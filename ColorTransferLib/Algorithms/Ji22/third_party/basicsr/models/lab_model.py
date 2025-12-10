import os
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from ..archs import build_network
from ..losses import build_loss
from ..metrics import calculate_metric
from ..utils import get_root_logger, imwrite, tensor2img
from ..utils.img_util import tensor_lab2rgb
from ..utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class LABModel(SRModel):
    """DynamicUnetFullGAN model for image colorization."""

    def init_training_settings(self):
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
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq_rgb = tensor_lab2rgb(torch.cat([self.lq, torch.zeros_like(self.lq), torch.zeros_like(self.lq)], dim=1))
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_lab = torch.cat([self.lq, self.gt], dim=1)
            self.gt_rgb = tensor_lab2rgb(self.gt_lab)

    def optimize_parameters(self, current_iter):

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output_ab = self.net_g(self.lq_rgb)
        self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
        self.output_rgb = tensor_lab2rgb(self.output_lab)

        l_g_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output_ab, self.gt)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix
        # perceptual loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output_rgb, self.gt_rgb)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        # gan loss
        fake_g_pred = self.net_d(self.output_ab)
        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)

        l_g_total += l_g_gan
        loss_dict['l_g_gan'] = l_g_gan

        l_g_total.backward()
        self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output_ab.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()

        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq_rgb.detach().cpu()
        out_dict['result'] = self.output_rgb.detach().cpu()
        # only for verbose
        self.output_lab_chroma = torch.cat([torch.ones_like(self.lq) * 50, self.output_ab], dim=1)
        self.output_rgb_chroma = tensor_lab2rgb(self.output_lab_chroma)
        out_dict['result_chroma'] = self.output_rgb_chroma.detach().cpu()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt_rgb.detach().cpu()
            # only for verbose
            self.gt_lab_chroma = torch.cat([torch.ones_like(self.lq) * 50, self.gt], dim=1)
            self.gt_rgb_chroma = tensor_lab2rgb(self.gt_lab_chroma)
            out_dict['gt_chroma'] = self.gt_rgb_chroma.detach().cpu()
        return out_dict

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output_ab = self.net_g_ema(self.lq_rgb)
                self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
                self.output_rgb = tensor_lab2rgb(self.output_lab)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output_ab = self.net_g(self.lq_rgb)
                self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
                self.output_rgb = tensor_lab2rgb(self.output_lab)
            self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        if with_metrics:
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            if idx == 100:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            if hasattr(self, 'gt'):
                del self.gt
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img

            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_dir = osp.join(self.opt['path']['visualization'], img_name)
                    for key in visuals:
                        save_path = os.path.join(save_dir, '{}_{}.png'.format(current_iter, key))
                        img = tensor2img(visuals[key])
                        imwrite(img, save_path)
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

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class LABGANRGBModel(LABModel):

    def optimize_parameters(self, current_iter):

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output_ab = self.net_g(self.lq_rgb)
        self.output_lab = torch.cat([self.lq, self.output_ab], dim=1)
        self.output_rgb = tensor_lab2rgb(self.output_lab)

        l_g_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output_ab, self.gt)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix
        # perceptual loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output_rgb, self.gt_rgb)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        # gan loss
        fake_g_pred = self.net_d(self.output_rgb)
        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)

        l_g_total += l_g_gan
        loss_dict['l_g_gan'] = l_g_gan

        l_g_total.backward()
        self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(self.gt_rgb)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output_rgb.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()

        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)