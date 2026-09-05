"""
im_channels.py
==============
三种索引调制（IM）物理层链路的纯 NumPy Monte Carlo 仿真：

- ``FAIMChannel``  : 流体天线索引调制（FA-IM），毫米波几何多径信道 +
                     容量最优端口选择 + ML 检测（对齐 TCOM SSC 论文参数）。
- ``SMChannel``    : 传统空间调制（SM），Nt 根固定天线选 1，
                     i.i.d. Rayleigh MIMO 信道 + ML 检测。
- ``OFDMIMChannel``: OFDM 频域索引调制（OFDM-IM），每组 n 个子载波激活 k 个，
                     子载波独立 Rayleigh 衰落 + 逐组 ML 检测。

所有信道暴露统一接口::

    transmit(indices, snr_db, mode='seim'|'eep', rng=None) -> indices_hat

其中 ``indices`` 是 (L, Nq) 的整数码本索引矩阵（取值 [0, Ne)），
``mode='seim'``  表示语义感知流分割（Semantic-aware Index Modulation），
``mode='eep'``   表示无语义意识的均等分割（对应 "SSC w/o SS" 基线）。

每个信道还提供 ``measure(snr_db, n_slots, rng)`` 用于离线统计
索引流 / 符号流的误比特率，从而决定哪一路物理流更鲁棒
（对应 TCOM 论文 Algorithm 1 的离线 Monte Carlo 预配置步骤）。
"""

import math
from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------------
# 比特工具
# ---------------------------------------------------------------------------
def decimal_to_bits(dec, nbits):
    """(...,) 整数 -> (..., nbits) 0/1 数组，高位在前。"""
    dec = np.asarray(dec, dtype=np.int64)
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.int64)
    return ((dec[..., None] >> shifts) & 1).astype(np.int64)


def bits_to_decimal(bits):
    """(..., nbits) 0/1 数组 -> (...,) 整数。"""
    bits = np.asarray(bits, dtype=np.int64)
    nbits = bits.shape[-1]
    weights = (1 << np.arange(nbits - 1, -1, -1, dtype=np.int64))
    return (bits * weights).sum(axis=-1)


def gray_qam_constellation(M):
    """单位平均功率、格雷映射的方型 QAM 星座。"""
    sqrt_m = int(round(math.sqrt(M)))
    assert sqrt_m * sqrt_m == M, "M 必须为完全平方数 (16/64/256...)"
    m_1d = int(math.log2(sqrt_m))
    gray = np.array([i ^ (i >> 1) for i in range(2 ** m_1d)])
    axis = np.arange(-(sqrt_m - 1), sqrt_m, 2, dtype=np.float64)
    const = np.zeros(M, dtype=np.complex128)
    for i in range(sqrt_m):
        for j in range(sqrt_m):
            const[i * sqrt_m + j] = axis[gray[i]] + 1j * axis[gray[j]]
    return const / np.sqrt(np.mean(np.abs(const) ** 2))


def ml_detect(y, table, batch=512):
    """
    最大似然检测: y (Nr, F), table (Nr, C) -> (F,) 候选索引。
    通过能量展开避免构造大三维数组: ||y - t||^2 = ||y||^2 + ||t||^2 - 2Re(y^H t)。
    """
    Nr, F = y.shape
    C = table.shape[1]
    table_energy = np.sum(np.abs(table) ** 2, axis=0)  # (C,)
    dec = np.empty(F, dtype=np.int64)
    for s in range(0, F, batch):
        yb = y[:, s:s + batch]                          # (Nr, b)
        y_energy = np.sum(np.abs(yb) ** 2, axis=0)      # (b,)
        cross = np.real(yb.conj().T @ table)            # (b, C)
        dist = y_energy[:, None] + table_energy[None, :] - 2.0 * cross
        dec[s:s + batch] = np.argmin(dist, axis=1)
    return dec


# ---------------------------------------------------------------------------
# 基类：统一的分割 / 传输 / 合并流程
# ---------------------------------------------------------------------------
class IMChannelBase:
    """所有 IM 链路的基类。子类只需实现 ``_tx_frames``。

    属性
    ----
    m1 : int   每时隙索引流比特数（log2 索引实体数）
    m2 : int   每时隙符号流比特数（log2 星座阶数 M）
    bits_per_index : int  码本索引的比特数（log2 Ne）
    index_more_robust : bool  索引流是否比符号流更鲁棒（由离线 measure 决定）
    """

    m1 = None
    m2 = None
    bits_per_index = 4
    index_more_robust = True
    name = "IM"

    # -- 子类实现：物理层传输，输入/输出均为 (F,) 十进制索引 ---------------
    def _tx_frames(self, port_idx, sym_idx, snr_db, rng):
        raise NotImplementedError

    # -- 语义感知流分割 vs 均等分割 ----------------------------------------
    def transmit(self, indices, snr_db, mode="seim", rng=None):
        rng = rng or np.random.default_rng()
        indices = np.asarray(indices, dtype=np.int64)
        L, Nq = indices.shape
        total = L * Nq
        split = int(round(total * self.m1 / (self.m1 + self.m2)))

        if mode == "seim":
            # 列优先展开: 索引序列按 RQ 步（语义重要性）降序排列
            f = indices.T.reshape(-1)
        elif mode == "eep":
            # 行优先展开: 各 RQ 步自然混合，无语义意识（"w/o SS" 基线）
            f = indices.reshape(-1)
        else:
            raise ValueError("mode 必须为 'seim' 或 'eep'")

        if mode == "seim" and not self.index_more_robust:
            # 符号流更鲁棒 -> 重要前缀分配给符号流
            sym_len = total - split
            index_src, symbol_src = f[sym_len:], f[:sym_len]
        else:
            # 索引流更鲁棒（或 eep）-> 前段分配给索引流
            index_src, symbol_src = f[:split], f[split:]

        # 十进制 -> 比特 -> 分帧（每时隙: m1 索引比特 + m2 符号比特）
        ibits = decimal_to_bits(index_src, self.bits_per_index).reshape(-1)
        sbits = decimal_to_bits(symbol_src, self.bits_per_index).reshape(-1)
        pad_i = (-len(ibits)) % self.m1
        pad_s = (-len(sbits)) % self.m2
        ibits = np.concatenate([ibits, np.zeros(pad_i, dtype=np.int64)])
        sbits = np.concatenate([sbits, np.zeros(pad_s, dtype=np.int64)])
        port_idx = bits_to_decimal(ibits.reshape(-1, self.m1))
        sym_idx = bits_to_decimal(sbits.reshape(-1, self.m2))
        F = max(len(port_idx), len(sym_idx))
        port_idx = np.pad(port_idx, (0, F - len(port_idx)))
        sym_idx = np.pad(sym_idx, (0, F - len(sym_idx)))

        # 物理层传输
        port_hat, sym_hat = self._tx_frames(port_idx, sym_idx, snr_db, rng)

        # 恢复比特流并截断补零
        ib_hat = decimal_to_bits(port_hat, self.m1).reshape(-1)[: len(ibits) - pad_i]
        sb_hat = decimal_to_bits(sym_hat, self.m2).reshape(-1)[: len(sbits) - pad_s]
        index_hat = bits_to_decimal(ib_hat.reshape(-1, self.bits_per_index))
        symbol_hat = bits_to_decimal(sb_hat.reshape(-1, self.bits_per_index))

        if mode == "seim" and not self.index_more_robust:
            f_hat = np.concatenate([symbol_hat, index_hat])
        else:
            f_hat = np.concatenate([index_hat, symbol_hat])

        if mode == "seim":
            return f_hat.reshape(Nq, L).T
        return f_hat.reshape(L, Nq)

    # -- 离线双流可靠性统计（Algorithm 1 的预配置步骤） ---------------------
    def measure(self, snr_db, n_slots, rng=None):
        """返回 (ber_index, ber_symbol)：索引流与符号流的误比特率。

        将时隙均分为 num_H 段分别传输（每次 _tx_frames 调用随机选取一个
        信道实现），使统计平均覆盖整个信道实现池，BER 曲线更光滑。
        """
        rng = rng or np.random.default_rng()
        n_rep = max(1, getattr(self, "num_H", 1))
        per = max(1, n_slots // n_rep)
        ber_i_list, ber_s_list = [], []
        for _ in range(n_rep):
            port_idx = rng.integers(0, 2 ** self.m1, size=per)
            sym_idx = rng.integers(0, 2 ** self.m2, size=per)
            port_hat, sym_hat = self._tx_frames(port_idx, sym_idx, snr_db, rng)
            ber_i_list.append(np.mean(decimal_to_bits(port_idx, self.m1)
                                      != decimal_to_bits(port_hat, self.m1)))
            ber_s_list.append(np.mean(decimal_to_bits(sym_idx, self.m2)
                                      != decimal_to_bits(sym_hat, self.m2)))
        return float(np.mean(ber_i_list)), float(np.mean(ber_s_list))


# ---------------------------------------------------------------------------
# FA-IM：流体天线索引调制（毫米波几何信道 + 容量最优端口选择）
# ---------------------------------------------------------------------------
class FAIMChannel(IMChannelBase):
    name = "FA-IM"

    def __init__(self, Ns=4, Np=16, Nr=8, M=64, W=2.0, L_paths=10,
                 num_H=8, Ne=16, seed=0):
        assert math.log2(Ns).is_integer() and math.log2(M).is_integer()
        self.Ns, self.Np, self.Nr, self.M = Ns, Np, Nr, M
        self.m1 = int(math.log2(Ns))
        self.m2 = int(math.log2(M))
        self.bits_per_index = int(math.log2(Ne))
        self.num_H = num_H
        rng = np.random.default_rng(seed)
        H_pool = self._mmwave_channel(rng, num_H, Nr, Np, W, L_paths)
        self.Hs_pool = np.stack([self._select_ports(H) for H in H_pool])  # (num_H,Nr,Ns)
        self.constellation = gray_qam_constellation(M)
        # 预计算全部候选发射信号与 ML 查找表
        all_x = np.zeros((Ns * M, Ns), dtype=np.complex128)
        for p in range(Ns):
            all_x[p * M:(p + 1) * M, p] = self.constellation
        self.lookup = np.stack([H @ all_x.T for H in self.Hs_pool])       # (num_H,Nr,Ns*M)

    @staticmethod
    def _mmwave_channel(rng, num, Nr, Np, W, L):
        dr, dt = 0.5, (W / (Np - 1) if Np > 1 else 0.0)
        rx = np.arange(Nr) * dr
        tx = np.arange(Np) * dt
        aod = rng.random((num, L)) * math.pi - math.pi / 2
        aoa = rng.random((num, L)) * math.pi - math.pi / 2
        g = (rng.standard_normal((num, L)) + 1j * rng.standard_normal((num, L))) / math.sqrt(2 * L)
        a_r = np.exp(2j * math.pi * np.sin(aoa)[..., None] * rx)   # (num,L,Nr)
        a_t = np.exp(2j * math.pi * np.sin(aod)[..., None] * tx)   # (num,L,Np)
        H = g[:, :, None, None] * a_r[:, :, :, None] * a_t.conj()[:, :, None, :]
        return H.sum(axis=1)                                       # (num,Nr,Np)

    def _select_ports(self, H):
        combos = np.array(list(combinations(range(self.Np), self.Ns)))   # (C,Ns)
        Hc = H[:, combos]                                                # (C,Nr,Ns)
        A = np.eye(self.Ns) + (Hc.conj().transpose(0, 2, 1) @ Hc) / self.Ns
        best = np.argmax(np.linalg.slogdet(A)[1])
        return H[:, combos[best]]

    def _tx_frames(self, port_idx, sym_idx, snr_db, rng):
        F = len(port_idx)
        iH = int(rng.integers(0, self.num_H))
        Hs, table = self.Hs_pool[iH], self.lookup[iH]
        s = self.constellation[sym_idx]
        y = Hs[:, port_idx] * s[None, :]                                 # (Nr,F)
        nv = 1.0 / (10.0 ** (snr_db / 10.0))
        y = y + math.sqrt(nv / 2) * (rng.standard_normal((self.Nr, F))
                                     + 1j * rng.standard_normal((self.Nr, F)))
        dec = ml_detect(y, table)
        return dec // self.M, dec % self.M


# ---------------------------------------------------------------------------
# SM：传统空间调制（Nt 选 1，i.i.d. Rayleigh MIMO）
# ---------------------------------------------------------------------------
class SMChannel(IMChannelBase):
    name = "SM"

    def __init__(self, Nt=4, Nr=4, M=64, num_H=8, Ne=16, seed=1):
        assert math.log2(Nt).is_integer() and math.log2(M).is_integer()
        self.Nt, self.Nr, self.M = Nt, Nr, M
        self.m1 = int(math.log2(Nt))
        self.m2 = int(math.log2(M))
        self.bits_per_index = int(math.log2(Ne))
        self.num_H = num_H
        rng = np.random.default_rng(seed)
        self.H_pool = (rng.standard_normal((num_H, Nr, Nt))
                       + 1j * rng.standard_normal((num_H, Nr, Nt))) / math.sqrt(2)
        self.constellation = gray_qam_constellation(M)
        all_x = np.zeros((Nt * M, Nt), dtype=np.complex128)
        for p in range(Nt):
            all_x[p * M:(p + 1) * M, p] = self.constellation
        self.lookup = np.stack([H @ all_x.T for H in self.H_pool])

    def _tx_frames(self, port_idx, sym_idx, snr_db, rng):
        F = len(port_idx)
        iH = int(rng.integers(0, self.num_H))
        H, table = self.H_pool[iH], self.lookup[iH]
        s = self.constellation[sym_idx]
        y = H[:, port_idx] * s[None, :]
        nv = 1.0 / (10.0 ** (snr_db / 10.0))
        y = y + math.sqrt(nv / 2) * (rng.standard_normal((self.Nr, F))
                                     + 1j * rng.standard_normal((self.Nr, F)))
        dec = ml_detect(y, table)
        return dec // self.M, dec % self.M


# ---------------------------------------------------------------------------
# OFDM-IM：频域（子载波）索引调制
# ---------------------------------------------------------------------------
class OFDMIMChannel(IMChannelBase):
    name = "OFDM-IM"

    def __init__(self, n=4, k=2, M=16, Ne=16, seed=2):
        self.n, self.k, self.M = n, k, M
        all_patterns = list(combinations(range(n), k))
        n_index_bits = int(math.floor(math.log2(len(all_patterns))))
        self.m1 = n_index_bits
        self.m2 = int(math.log2(M)) * k          # 每组符号比特（k 个活跃子载波）
        self.bits_per_index = int(math.log2(Ne))
        # 标准做法：仅使用 2^m1 个激活图案
        self.patterns = np.array(all_patterns[: 2 ** n_index_bits])      # (P,k)
        self.constellation = gray_qam_constellation(M)
        # 预计算每组全部候选向量: P * M^k 个
        P = len(self.patterns)
        cand = np.zeros((P * M * M, n), dtype=np.complex128)
        row = 0
        for p in range(P):
            for s1 in range(M):
                for s2 in range(M):
                    cand[row, self.patterns[p]] = [self.constellation[s1],
                                                   self.constellation[s2]]
                    row += 1
        self.cand = cand
        self.n_cand = P * M * M

    def _tx_frames(self, port_idx, sym_idx, snr_db, rng):
        """port_idx: (F,) 激活图案索引; sym_idx: (F,) 联合符号索引（两个 M 进制符号
        合并为 M^2 进制: s = s1*M + s2）。"""
        F = len(port_idx)
        s1, s2 = sym_idx // self.M, sym_idx % self.M
        flat = port_idx * (self.M * self.M) + s1 * self.M + s2           # (F,)
        x = self.cand[flat]                                              # (F,n)
        # 每个子载波独立 Rayleigh 衰落，接收端已知
        h = (rng.standard_normal((F, self.n))
             + 1j * rng.standard_normal((F, self.n))) / math.sqrt(2)
        nv = 1.0 / (10.0 ** (snr_db / 10.0))
        noise = math.sqrt(nv / 2) * (rng.standard_normal((F, self.n))
                                     + 1j * rng.standard_normal((F, self.n)))
        y = h * x + noise
        # 逐组 ML：dist = ||y - h * cand||^2
        dec = np.empty(F, dtype=np.int64)
        B = 4096
        for s in range(0, F, B):
            hx = h[s:s + B, None, :] * self.cand[None, :, :]             # (b,C,n)
            d = np.abs(y[s:s + B, None, :] - hx) ** 2
            dec[s:s + B] = np.argmin(d.sum(axis=-1), axis=1)
        p_hat = dec // (self.M * self.M)
        rem = dec % (self.M * self.M)
        sym_hat = (rem // self.M) * self.M + (rem % self.M)
        return p_hat, sym_hat

    # OFDM-IM 每时隙的"符号比特"是 m2（=k*log2 M），transmit/measure 的
    # 通用比特换算不受影响；但 sym_idx 的动态范围是 M^k，需覆盖 measure。
    def measure(self, snr_db, n_slots, rng=None):
        rng = rng or np.random.default_rng()
        port_idx = rng.integers(0, 2 ** self.m1, size=n_slots)
        sym_idx = rng.integers(0, self.M ** self.k, size=n_slots)
        port_hat, sym_hat = self._tx_frames(port_idx, sym_idx, snr_db, rng)
        ber_i = np.mean(decimal_to_bits(port_idx, self.m1)
                        != decimal_to_bits(port_hat, self.m1))
        ber_s = np.mean(decimal_to_bits(sym_idx, self.m2)
                        != decimal_to_bits(sym_hat, self.m2))
        return ber_i, ber_s


def decide_robust_stream(channel, snr_list, n_slots=20000, seed=123):
    """对给定信道在整个工作 SNR 区间统计双流误比特率，
    返回 (index_more_robust, ber_i_list, ber_s_list)。"""
    rng = np.random.default_rng(seed)
    ber_i, ber_s = [], []
    for snr in snr_list:
        bi, bs = channel.measure(snr, n_slots, rng)
        ber_i.append(bi)
        ber_s.append(bs)
    index_more_robust = float(np.sum(ber_i)) <= float(np.sum(ber_s))
    return index_more_robust, ber_i, ber_s
