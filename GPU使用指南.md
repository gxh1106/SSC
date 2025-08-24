# GPU 使用指南

## hugging face

### 0 配置环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 1 下载模型

```bash
huggingface-cli download --resume-download [model name] --local-dir [file path]
# modelscope 下载无需任何配置
modelscope download --model [model name] --local_dir [./qwen2.5-3b]
```

### 2 下载数据集

```bash
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
```

### 3 从本地加载tokenizer

```bash
tokenizer = AutoTokenizer.from_pretrained("/mnt/nvme0n1/workspace/gxh/uer/roberta-base-finetuned-dianping-chinese/")
```

## 常用指令

### 1. 使用screen方法创建线程

```bash
screen -R name #新建/打开名为name的进程（不会重名）
screen -S name #新建名为name的进程（会重名）
screen -ls #列出所有进程
ctrl + A,D #返回
exit #删除
# screen -R [Name] -X quit #删除进程
```

### 2. 查看gpu状态

```bash
gpustat -i
ctrl + c #结束
```

## 连接服务器

1. [ZeroTier | Global Networking Solution for IoT, SD-WAN, and VPN](https://www.zerotier.com/)创建账号，登陆，并下载软件

2. 在zerotier软件中`join new network`连接`56374ac9a4f917d3`网络，并告知管理员，管理员会将此设备移入网络

   <img src="https://xiaoqixiaowei.oss-cn-chengdu.aliyuncs.com/img_for_typora/image-20241104121043731.png" alt="image-20241104121043731" style="zoom:33%;" />

3. 至此即可用vscode的ssh连接至该GPU服务器
gxh
gxh123

4. vscode的账号是192.168.192.199，e.g. xxx@192.168.192.199

## GPU使用指南

==注意自己的所有程序都放在/data/xxx目录下，不要把自己的程序放在home目录下，如果data下没有自己的名字的文件夹，请自行创建，如果创建的时候有权限限制的，通知管理员==

1. 安装anaconda环境

[如何在Linux服务器上安装Anaconda（超详细）_linux安装anconda-CSDN博客](https://blog.csdn.net/wyf2017/article/details/118676765)

[安装anaconda到指定目录教程（Linux系统）_linux安装anaconda到指定目录-CSDN博客](https://blog.csdn.net/weixin_43120985/article/details/118163799)

注意已经有sh文件到/mnt/nvme0n1/workspace/xxx的目录下，如果没有的话拖动到这个目录下

```bash
chmod +x Anaconda3-5.3.0-Linux-x86_64.sh
```

```bash
./Anaconda3-5.3.0-Linux-x86_64.sh
```

```bash
一直enter到yes
然后系统提示是否是默认安装地址，自己输入安装地址eg. /mnt/nvme0n1/workspace/zcx/anaconda3
```

```bash
直到安装vscode的时候选择no，重启vscode
```

```bash
conda info -e
```

```bash
conda activate base
```

2. 查看gpu占用情况

   ```bash
   gpustat -i
   ```

   ctrl C 退出

3. 设置密钥（免密登录）

```bash
https://blog.csdn.net/Zhangye1011/article/details/141133327
```

4. 复制别人的环境

```bash
conda create -n pytorch_py38 --clone /data/zcx/anaconda3/envs/GS
```

5. 使用screen方法创建线程

![image-20250425211704837](https://xiaoqixiaowei.oss-cn-chengdu.aliyuncs.com/img_for_typora/20250425211704920.png)

![image-20250425211543739](https://xiaoqixiaowei.oss-cn-chengdu.aliyuncs.com/img_for_typora/20250425211549952.png)

![image-20250425211553983](https://xiaoqixiaowei.oss-cn-chengdu.aliyuncs.com/img_for_typora/20250425211554096.png)

用户名修改为![image-20250425211620022](https://xiaoqixiaowei.oss-cn-chengdu.aliyuncs.com/img_for_typora/20250425211620107.png)

![image-20250425211731140](https://xiaoqixiaowei.oss-cn-chengdu.aliyuncs.com/img_for_typora/20250425211731239.png)

输入密码，问管理员，注意需要交大vpn

![image-20250425211755561](https://xiaoqixiaowei.oss-cn-chengdu.aliyuncs.com/img_for_typora/20250425211755663.png)
