from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _gather_feat_gridneighbor(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(3).expand(ind.size(0), ind.size(1),ind.size(2), dim//ind.size(2)).contiguous().view(ind.size(0), ind.size(1), -1)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _tranpose_and_gather_feat_gridneighbor(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat_gridneighbor(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def landmark_flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 68, 2,
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def clothlandmark_flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 294, 2,
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def gaussian_fit(X, Y, Z):
    X = X.type(torch.FloatTensor).cuda()
    Y = Y.type(torch.FloatTensor).cuda()
    x4 = (X ** 4).sum()
    x2y2 = (X ** 2 * Y ** 2).sum()
    x3 = (X ** 3).sum()
    x2y = (X ** 2 * Y).sum()
    x2 = (X ** 2).sum()
    y4 = (Y ** 4).sum()
    xy2 = (X * Y ** 2).sum()
    y3 = (Y ** 3).sum()
    y2 = (Y ** 2).sum()
    xy = (X * Y).sum()
    x = (X).sum()
    y = (Y).sum()
    c = X.size(0)

    B = torch.Tensor([[x4, x2y2, x3, x2y, x2],
                  [x2y2, y4, xy2, y3, y2],
                  [x3, xy2, x2, xy, x],
                  [x2y, y3, xy, y2, y],
                  [x2, y2, x, x, c]]).cuda()

    lnf = torch.log(Z).type_as(B)
    z1= (X ** 2 * lnf).sum()
    z2 = (Y ** 2 * lnf).sum()
    z3 = (X * lnf).sum()
    z4 = (Y * lnf).sum()
    z5 = (lnf).sum()

    B_test = B.clone().cpu().numpy()
    B_rank = np.linalg.matrix_rank(B_test)
    if B_rank == 5:
       B = torch.inverse(B)
       K1 = B[0][0] *z1 + B[0][1] *z2 +  B[0][2] *z3 + B[0][3] *z4 +  B[0][4] *z5
       K2 = B[1][0] * z1 + B[1][1] * z2 + B[1][2] * z3 + B[1][3] * z4 + B[1][4] * z5
       K3 = B[2][0] * z1 + B[2][1] * z2 + B[2][2] * z3 + B[2][3] * z4 + B[2][4] * z5
       K4 = B[3][0] * z1 + B[3][1] * z2 + B[3][2] * z3 + B[3][3] * z4 + B[3][4] * z5
       K5 = B[4][0] * z1 + B[4][1] * z2 + B[4][2] * z3 + B[4][3] * z4 + B[4][4] * z5

       sigma_w2 = -0.5 / (K1+1e20)
       sigma_h2 = -0.5 / (K2+1e20)
       mu_w = K3 * (-0.5/(K1+1e20))
       mu_h = K4 * (-0.5/(K2+1e20))
       return [sigma_w2,sigma_h2,mu_w,mu_h]
    else:
        return [0, 0, 0, 0]