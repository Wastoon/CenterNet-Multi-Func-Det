# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat, _tranpose_and_gather_feat_gridneighbor
import torch.nn.functional as F
from .utils import gaussian_fit


def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def multi_gaussian_Kl_divergence(outsigmaW, outsigmaH, outmuW, outmuH, sigmaW_gt, sigmaH_gt):
    kl_divergience = [0.5 * torch.log((sigmaW_gt[i] ** 2 * sigmaH_gt[i] ** 2) / (outsigmaW[i] * outsigmaH[i])) - 1
                      + 0.5 * ((outsigmaW[i] / sigmaW_gt[i] ** 2) + (outsigmaH[i] / sigmaH_gt[i] ** 2)) +
                      0.5 * ((outmuW[i] ** 2 / sigmaW_gt[i] ** 2) + (outmuH[i] ** 2 / sigmaH_gt[i] ** 2))
                      for i in range(len(outsigmaW))]
    return sum(kl_divergience)


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class Focalloss_exphm_and_sigma_KL_divergence(nn.Module):
    def __init__(self):
        super(Focalloss_exphm_and_sigma_KL_divergence, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, hm_out, hm_gt, ct_ind, sigma_wh, hm_mask, sigmawh_mask):
        """
        hm_gt : batch * clsnum * output_size * output_size
        hm_out : batch * clsnum * output_size * output_size
        ct_ind : batch * clsnum * maxobjsnum * 2
        sigma_wh : batch * clsnum * maxobjs * 2
        regmask: batch * maxobjs * 2
        hm_mask : batch * clsnum * output_size * output_size
        sigmawh_mask: batch * clsnum * maxobjs
        """
        #focal_loss_norm = self.neg_loss(torch.clamp((hm_out * hm_mask).sigmoid_(), min=1e-4, max=1 - 1e-4), hm_gt * hm_mask )
        focal_loss_norm = self.neg_loss(torch.clamp((hm_out * hm_mask), min=1e-6, max=1 - 1e-6), hm_gt * hm_mask)
        #focal_loss_norm = torch.nn.MSELoss()(torch.clamp((hm_out * hm_mask), min=1e-6, max=1 - 1e-6), hm_gt * hm_mask)
        #focal_loss_norm = self.neg_loss(torch.clamp((hm_out* hm_mask).sigmoid_(), min=1e-4, max=1 - 1e-4),hm_gt * hm_mask)
        focal_loss_norm = self.neg_loss(torch.clamp((hm_out).sigmoid_(), min=1e-6, max=1 - 1e-6), hm_gt * hm_mask)
        #import pdb
        #db.set_trace()
        batch_channel_sigma_w_list = []
        batch_channel_sigma_h_list = []
        batch_channel_mu_w_list = []
        batch_channel_mu_h_list = []
        gt_channel_sigma_w_list = []
        gt_channel_sigma_h_list = []

        for batch in range(hm_out.size(0)):
            for cls in range(hm_out.size(1)):
                if int(sigmawh_mask[batch][cls].sum()) > 0:
                    for obj in range(int(sigmawh_mask[batch][cls].sum())):
                        if int(sigmawh_mask[batch][cls][obj]) == 1:
                            x, y = int(ct_ind[batch][cls][obj][0]), int(ct_ind[batch][cls][obj][1])
                            sigma_w, sigma_h = sigma_wh[batch][cls][obj]
                            radius_w, radius_h = int((sigma_w * 6 - 1) / 2), int((sigma_h * 6 - 1) / 2)
                            height, width = hm_out.shape[2:]
                            left, right = int(min(x, radius_w)), int(min(width - x, radius_w + 1))
                            top, bottom = int(min(y, radius_h)), int(min(height - y, radius_h + 1))

                            peer_objs_hm_pdfvalue = hm_out.clone()[batch][cls][y - top:y + bottom,
                                                    x - left:x + right].contiguous().view(-1)
                            ys, xs = torch.meshgrid((torch.arange(-1 * radius_h, radius_h + 1, 1),
                                                     torch.arange(-1 * radius_w, radius_w + 1, 1)))
                            xs = xs.reshape(-1).type_as(peer_objs_hm_pdfvalue).cuda()
                            ys = ys.reshape(-1).type_as(peer_objs_hm_pdfvalue).cuda()
                            out_sigmaw, out_sigmah, out_muw, out_muh = gaussian_fit(xs, ys, peer_objs_hm_pdfvalue)

                            if out_sigmaw >0 and out_sigmah >0:
                                batch_channel_sigma_w_list.append(out_sigmaw)
                                batch_channel_sigma_h_list.append(out_sigmah)
                                batch_channel_mu_w_list.append(out_muw)
                                batch_channel_mu_h_list.append(out_muh)
                                gt_channel_sigma_w_list.append(sigma_w)
                                gt_channel_sigma_h_list.append(sigma_h)
        if len(batch_channel_sigma_w_list) != 0:
            Kl_divergence = multi_gaussian_Kl_divergence(batch_channel_sigma_w_list, batch_channel_sigma_h_list,
                                                         batch_channel_mu_w_list, batch_channel_mu_h_list,
                                                         gt_channel_sigma_w_list, gt_channel_sigma_h_list)

            return Kl_divergence , focal_loss_norm
        else:
            return None, focal_loss_norm


class RegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegL1loss_gridneighbor(nn.Module):
    def __init__(self):
        super(RegL1loss_gridneighbor, self).__init__()

    def forward(self, output, mask, ind, target, mainpoints_list):
        ##pred: Batch x max_objs x mainpoinsnum*2
        ##output: Batch x mainpoinsnum*2 x w x h
        ##ind: Batch x max_objs x mainpoinsnum
        ##target: Batch x max_objs x 2

        pred = _tranpose_and_gather_feat_gridneighbor(output, ind)
        mask = mask.unsqueeze(3).expand(mask.size(0), mask.size(1), mask.size(2),
                                        pred.size(2) // mask.size(2)).contiguous().view(mask.size(0), mask.size(1),
                                                                                        -1).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = 0
        for point in range(len(mainpoints_list)):
            loss += F.l1_loss(pred[:, :, point:point + 2] * mask[:, :, point:point + 2],
                              target[:, :, point:point + 2] * mask[:, :, point:point + 2], size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
