from loss.distortion import Distortion
from random import choice
import torch.nn as nn

import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from copy import deepcopy

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel


@MODEL_REGISTRY.register()
class SSCModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SSCModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key, frozen_encoder=self.opt['network_g'].get('frozen_encoder', False))

        if self.is_train:
            self.init_training_settings()

    def load_network(self, net, load_path, strict=True, param_key='params', frozen_encoder=False):
        """修改后的网络加载函数.

        此函数可以灵活加载权重，能够处理某些层（如全连接层）尺寸不匹配
        以及特定缓存（如attn_mask）需要被忽略的情况。

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): 在此实现中，主要通过手动比较来控制加载，此参数作用有限。
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            # 兼容不带params键的权重文件
            if param_key in load_net:
                load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        
        # 移除不必要的 'module.' 前缀
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)

        # 1. 移除所有 attn_mask，避免尺寸强行匹配错误
        filtered_load_net = {}
        removed_keys = []
        for k, v in load_net.items():
            if "attn_mask" in k:
                removed_keys.append(k)
            else:
                filtered_load_net[k] = v
        if removed_keys:
            logger.info(f'Removed keys containing "attn_mask": {removed_keys}')

        # 2. 准备加载最终匹配的权重
        model_state_dict = net.state_dict()
        final_load_dict = {}
        mismatched_shape_keys = []
        
        for k, v in filtered_load_net.items():
            if k in model_state_dict:
                # 只加载名称和形状都匹配的权重
                if model_state_dict[k].shape == v.shape:
                    final_load_dict[k] = v
                else:
                    mismatched_shape_keys.append(
                        f"{k} (pretrained shape: {v.shape}, model shape: {model_state_dict[k].shape})")
        
        if mismatched_shape_keys:
            logger.warning(f'Skipped loading keys due to shape mismatch: {mismatched_shape_keys}')

        # 3. 使用 strict=False 加载，以处理新模型中存在但预训练模型中没有的层
        # (例如，如果您添加了全新的层)
        missing_keys, unexpected_keys = net.load_state_dict(final_load_dict, strict=False)

        if missing_keys:
            logger.info(f'Keys not found in pre-trained model (kept random init): {missing_keys}')
        # unexpected_keys 应该为空，因为我们已经筛选过了，但为了安全起见仍然打印
        if unexpected_keys:
            logger.warning(f'Keys from pre-trained model not found in new model: {unexpected_keys}')

        if frozen_encoder:
            logger.info("Freezing encoder parameters as requested...")
            # 检查网络是否真的有 encoder 这个子模块，增加代码的健壮性
            if hasattr(net, 'encoder'):
                for param in net.encoder.parameters():
                    param.requires_grad = False
                logger.info("All parameters in 'net.encoder' have been frozen.")
            else:
                logger.warning(f"Attempted to freeze encoder, but network '{net.__class__.__name__}' has no 'encoder' attribute.")

    # def load_network(self, net, load_path, strict=True, param_key='params'):
    #     """Load network.

    #     Args:
    #         load_path (str): The path of networks to be loaded.
    #         net (nn.Module): Network.
    #         strict (bool): Whether strictly loaded.
    #         param_key (str): The parameter key of loaded network. If set to
    #             None, use the root 'path'.
    #             Default: 'params'.
    #     """
    #     logger = get_root_logger()
    #     net = self.get_bare_model(net)
    #     load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    #     if param_key is not None:
    #         if param_key not in load_net and 'params' in load_net:
    #             param_key = 'params'
    #             logger.info('Loading: params_ema does not exist, use params.')
    #         load_net = load_net[param_key]
    #     logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
    #     # remove unnecessary 'module.'
    #     for k, v in deepcopy(load_net).items():
    #         if k.startswith('module.'):
    #             load_net[k[7:]] = v
    #             load_net.pop(k)
    #     self._print_different_keys_loading(net, load_net, strict)
    #     net.load_state_dict(load_net, strict=strict)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema', frozen_encoder=self.opt['network_g'].get('frozen_encoder', False))
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.commit_loss_weight = train_opt.get('commit_loss_weight', 0.25)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.input = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output, CBR, chan_param, loss_commit, embed_idxs = self.net_g(self.input)

        l_total = self.commit_loss_weight * loss_commit
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.input)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.input)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, CBR, chan_param, loss_commit, embed_idxs = self.net_g_ema(self.input, given_SNR=10)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, CBR, chan_param, loss_commit, embed_idxs = self.net_g(self.input, given_SNR=10)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            gt_img = tensor2img([visuals['gt']])
            metric_data['img2'] = gt_img

            # tentative for out of GPU memory
            del self.input
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.input.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

