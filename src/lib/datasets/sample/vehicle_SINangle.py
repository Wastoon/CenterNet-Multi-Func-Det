from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
from math import tau

def DFT(t, coef_list, order=5):
    kernel = np.array([np.exp(-n*1j*t) for n in range(-order, order+1)])
    series = np.sum( (coef_list[:,0]+1j*coef_list[:,1]) * kernel[:])
    return np.real(series), np.imag(series)

class VehicleDetSINANGLEDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0] - box[2]/2, box[1] - box[3]/2, box[0] + box[2]/2, box[1] + box[3]/2],dtype=np.float32)
        ang = float(box[4])
        return bbox, ang

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):

        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32) ## [center_x, center_y]
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        rot = 0  ## write here just for val
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.aug_rot:
                rf = self.opt.rotate
                rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
                ##rot =30
                print(rot)
                rot_rad = rot / 180 * (np.pi)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, rot, [input_w, input_h])

        inp = cv2.warpAffine(img, trans_input,(input_w, input_h),flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output_rot = get_affine_transform(c, s, rot, [output_w, output_h])
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        #translation_x, translation_y = trans_output[0, 2], trans_output[1, 2]
        scale_ratio_x, scale_ratio_y = trans_output[0, 0], trans_output[1, 1]

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
        if self.opt.use_contour_fourier:
            contour = np.zeros((self.max_objs, (2*self.order+1)*2), dtype=np.float32)


        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox, an = self._coco_box_to_bbox(ann['bbox'])
            bbox_for_calcute_wh = bbox.copy()
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output_rot)
            bbox[2:] = affine_transform(bbox[2:], trans_output_rot)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            bbox_for_calcute_wh[:2] = affine_transform(bbox_for_calcute_wh[:2], trans_output)
            bbox_for_calcute_wh[2:] = affine_transform(bbox_for_calcute_wh[2:], trans_output)
            h, w = bbox_for_calcute_wh[3] - bbox_for_calcute_wh[1], bbox_for_calcute_wh[2] - bbox_for_calcute_wh[0]
            if (h > 0 and w > 0) or (rot != 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.opt.mse_loss else radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)

                wh[k] = 1. * w, 1. * h
                Angle_system_an = an /180 * (np.pi)
                if rot !=0:
                    Angle_system_an = Angle_system_an - rot_rad ## rot_rad :clockwise(shunShizhen:-)
                ang[k] = np.sin(Angle_system_an/2)
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.use_contour_fourier:
                    coef = np.array(ann['contour_coef'])
                    org_order = coef.shape[0]//2
                    coef = coef[org_order-self.order: org_order+self.order+1, :]

                    coef[:, 0] = coef[:, 0]* scale_ratio_x
                    coef[:, 1] = coef[:, 1]* scale_ratio_y

                    if self.opt.use_fourier_rotate:
                        #coef_temp = coef.copy()
                        rotate_angle = -1*Angle_system_an
                        if rot!=0:
                            rotate_angle = rotate_angle
                        rotate_factor = np.array([np.exp(1j*rotate_angle) for i in range(2*self.opt.fourier_order+1)])
                        coef_rotated = (coef[:,0]+1j*coef[:,1]) * rotate_factor[:]
                        coef[:, 0] = coef_rotated.real
                        coef[:, 1] = coef_rotated.imag

                    #coef[self.opt.fourier_order,:] = coef[self.opt.fourier_order,:] + np.array([translation_x, translation_y])

                    coef = coef.reshape(-1)
                    contour[k] = coef


                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2,
                                Angle_system_an, 1, cls_id])

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'ang':ang}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.use_contour_fourier:
            ret.update({'contour': contour})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 7), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id, 'file_name': file_name}
            if self.opt.use_contour_fourier:
                meta['contour'] = contour
        else:
            meta = {'c': c, 's': s, 'img_id': img_id, 'file_name': file_name}
        ret['meta'] = meta
        return ret