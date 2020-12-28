from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, RegL1loss_gridneighbor
from models.decode import vehicledet_decode, vehicledet_fourier_contour_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
from models.decode import fourier_contour_decode, de_scale_fourier_coef, de_translation_fourier_coef, DFT, reproduce_shape_by_fourier, decode_batch_curve, trans_curve2_obj_center, decode_batch_curve_gt
import cv2

class VehicledetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(VehicledetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg

        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt

        hm_loss, wh_loss, off_loss, ang_loss = 0, 0, 0, 0
        if opt.use_contour_fourier:
            contour_loss = 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                                       self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                    batch['dense_wh'] * batch['dense_wh_mask']) /
                                       mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'],
                        batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                else:
                    print(output['wh'].max())
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.ang_weight >0:
                output['ang'] = _sin(output['ang'])
                ang_loss += self.crit_reg(output['ang'], batch['reg_mask'], batch['ind'], batch['ang']) / opt.num_stacks

            if opt.use_contour_fourier and opt.fourier_contour_weight > 0:
                contour_loss += self.crit_reg(output['contour'],batch['reg_mask'], batch['ind'], batch['contour']) / opt.num_stacks

            if opt.use_contour_fourier:
                loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
                       opt.off_weight * off_loss + opt.ang_weight * ang_loss + opt.fourier_contour_weight * contour_loss
                loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                              'wh_loss': wh_loss, 'off_loss': off_loss, 'ang_loss': ang_loss, 'contour_loss':contour_loss}
            else:
                loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
                       opt.off_weight * off_loss + opt.ang_weight * ang_loss
                loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                              'wh_loss': wh_loss, 'off_loss': off_loss, 'ang_loss': ang_loss}

        return loss, loss_stats

def _relu(x):
    y = torch.clamp(x.relu_(), min = 0., max=179.99)
    return y

def _sin(x):
    y = torch.sin(x/2)
    return y

class VehicledetSINAngleTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(VehicledetSINAngleTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        if opt.use_contour_fourier:
            loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'ang_loss', 'contour_loss']
            loss = VehicledetLoss(opt)
        else:
            loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'ang_loss']
            loss = VehicledetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):

        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        ang = output['ang']#.relu_()
        ang = _sin(ang)
        ang = torch.asin(ang)*2
        #dets = vehicledet_decode(
        #    output['hm'], output['wh'], ang, reg=reg,
        #    cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets, center_xy_coordinate, inds = vehicledet_fourier_contour_decode(output['hm'], output['wh'],
                                                                             ang, reg=reg, cat_spec_wh=opt.cat_spec_wh,
                                                                             K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio

        print(batch['meta']['gt_det'].numpy().shape)
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        if opt.use_contour_fourier:
            ##1. center_xy up scale to output_res:
            center_xy_coordinate[:,:,:2] *= opt.down_ratio
            ##2. extract_contour_coef from heatmap and reback the curve of output_res(batch x K x time_length x 2)
            output_res_curves = decode_batch_curve(output['contour'].clone().cpu().detach(), inds.clone().cpu(), opt.down_ratio,
                               order=opt.fourier_order,
                               time_length=opt.fourier_time_length,
                               K=opt.K)
            ##3. trans the center from (0,0) to objs_center
            output_res_curves = trans_curve2_obj_center(output_res_curves, center_xy_coordinate.clone().cpu().detach()) ## (batch x K x time_length x 2)
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 5] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 5], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 5] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 5], img_id='out_gt')
            debugger.add_img(img, img_id='angled_pred_out')
            for k in range(len(dets[i])):
                if dets[i, k, 5] > opt.center_thresh:
                    debugger.add_coco_Rotate_bbox(dets[i, k, :4], dets[i, k, -1],dets[i, k, 4],
                                           dets[i, k, 5], img_id='angled_pred_out', system= 'Rad')

            debugger.add_img(img, img_id='angled_gt_out')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 5] > opt.center_thresh:
                    debugger.add_coco_Rotate_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],dets_gt[i, k, 4],
                                           dets_gt[i, k, 5], img_id='angled_gt_out', system= 'Rad')

            if opt.use_contour_fourier:
                debugger.add_img(img, img_id='fourier_dft_pred')
                for k in range(len(dets[i])):
                    if dets[i, k, 5] > opt.center_thresh:
                        debugger.add_dft_points_on_img(dets[i, k, -1], output_res_curves[i,k], img_id='fourier_dft_pred')
                contour_gt = batch['meta']['contour'].numpy().reshape(1, -1, (2*opt.fourier_order+1)*2) ###batch x max_objs x 402



                debugger.add_img(img, img_id='fourier_dft_gt')
                ##1. center_xy up scale to output_res:
                cx = torch.tensor((dets_gt[:, :, 2] + dets_gt[:, :, 0]) /2).unsqueeze(2)
                cy = torch.tensor((dets_gt[:, :, 3] + dets_gt[:, :, 1]) /2).unsqueeze(2)
                center_xy_coordinate_gt = torch.cat([cx, cy], dim=2)
                ##2. extract_contour_coef from heatmap and reback the curve of output_res(batch x K x time_length x 2)
                output_res_curves_gt = decode_batch_curve_gt(contour_gt,
                                                              opt.down_ratio,
                                                              order=opt.fourier_order,
                                                              time_length=opt.fourier_time_length,
                                                              K=center_xy_coordinate_gt.shape[1])
                ##3. trans the center from (0,0) to objs_center
                output_res_curves_gt = trans_curve2_obj_center(output_res_curves_gt,
                                                               center_xy_coordinate_gt.clone().cpu())  ## (batch x K x time_length x 2)

                for k in range(len(dets_gt[i])):
                    if dets_gt[i, k, 5] > opt.center_thresh:
                        debugger.add_dft_points_on_img(dets_gt[i, k, -1], output_res_curves_gt[i,k], img_id='fourier_dft_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=False)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = vehicledet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]