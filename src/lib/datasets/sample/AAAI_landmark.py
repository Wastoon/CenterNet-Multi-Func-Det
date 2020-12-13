import numpy as np
import cv2
import os
import torch.utils.data as data
import math
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from opts import opts
import matplotlib.pyplot as plt
class CenterLandmarkDataset(data.Dataset):

    def get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        #loadImgs(ids=[img_id]) return a list, whose length = 1
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        cropped = False
        if self.split == 'train':
            if np.random.random() < 1:
                cropped = True
                file_name = file_name.split('.')[0]+'crop.jpg'
                img_path = os.path.join(self.img_dir, file_name)
        if self.split == 'val':
            cropped = True

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        flipped = False
        rotted = False

        # input_res is max(input_h, input_w), input is the size of original img
        if np.random.random() < self.opts.keep_inp_res_prob and max((height | 127) + 1, (width | 127)  + 1) < 1024:
            self.opts.input_h = (height | 127) + 1
            self.opts.input_w = (width | 127)  + 1
            self.opts.output_h = self.opts.input_h // self.opts.down_ratio
            self.opts.output_w = self.opts.input_w // self.opts.down_ratio
            self.opts.input_res = max(self.opts.input_h, self.opts.input_w)
            self.opts.output_res = max(self.opts.output_h, self.opts.output_w)

        trans_input = get_affine_transform(
            c, s, rot, [self.opts.input_res, self.opts.input_res])
        inp = cv2.warpAffine(img, trans_input,
                             (self.opts.input_res, self.opts.input_res),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        #change data shape to [3, input_size, input_size]
        inp = inp.transpose(2, 0, 1)

        #output_res is max(output_h, output_w), output is the size after down sampling
        output_res = self.opts.output_res
        num_joints = self.num_joints
        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
        trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

        hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
        hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
        dense_kps = np.zeros((num_joints, 2, output_res, output_res), dtype=np.float32)
        dense_kps_mask = np.zeros((num_joints, output_res, output_res), dtype=np.float32)

        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        kps = np.zeros((self.max_objs, 2*num_joints), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.opts.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            if cropped:
                bbox = np.array(ann['bbox'])
            else:
                bbox = np.array(ann['org_bbox'])
            cls_id = int(ann['category_id']) - 1
            if cropped:
                pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
            else:
                pts = np.array(ann['org_keypoints'], np.float32).reshape(num_joints, 3)
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                for joint_idx in self.flip_idx:
                    pts[joint_idx[0]], pts[joint_idx[1]] = pts[joint_idx[1]].copy(), pts[joint_idx[0]].copy()#don't forget copy first
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            if rotted:
                pts_rot = np.zeros((num_joints, 2))
                for j in range(num_joints):
                    if pts[j, 2] > 0:
                        pts_rot[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                bbox[:2] = np.min(pts_rot, axis=0)
                bbox[2:] = np.max(pts_rot, axis=0)
            bbox = np.clip(bbox, 0, output_res - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if (h > 0 and w > 0) or (rot != 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = self.opts.hm_gauss if self.opts.mse_loss else max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_res + ct_int[0]
                reg[k] = ct - ct_int # the error of center[x, y]
                reg_mask[k] = 1
                num_kpts = pts[:, 2].sum() #whether joint can be seen or not
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                    reg_mask[k] = 0 #means this obj can'e be seen

                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)), min_overlap=1)
                hp_radius = self.opts.hm_gauss if self.opts.mse_loss else max(0, int(hp_radius))
                for j in range(num_joints):
                    if pts[j, 2] > 0:#means this joint can be seen
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if pts[j, 0] >= 0 and pts[j, 0] < output_res and pts[j, 1] >= 0 and pts[j, 1] < output_res:
                            kps[k, j*2: j*2+2] = pts[j, :2] - ct_int
                            kps_mask[k, j*2: j*2+2] = 1
                            pt_int = pts[j, :2].astype(np.int32)
                            hp_offset[k* num_joints +j] = pts[j, :2] - pt_int
                            hp_ind[k* num_joints + j] = pt_int[1] * output_res + pt_int[0]
                            hp_mask[k* num_joints + j] = 1
                            if self.opts.dense_hp:
                                # must be before draw center hm gaussian
                                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                                               pts[j, :2] - ct_int, radius, is_offset=True)
                                draw_gaussian(dense_kps_mask[j], ct_int, radius)
                            hp1 = draw_gaussian(hm_hp[j], pt_int, hp_radius)
                            # plt.imsave('/home/mry/Desktop/testimg/hp_{}_{}.jpg'.format(k, j), hp1)
                draw_gaussian(hm[cls_id], ct_int, radius)
                ##ge_det:x0, y0, x1, y1, joint1_x, joint1_y,...,joint17_x, joint17_y, cls_id
                gt_det.append([ct[0] - w/2, ct[1] - h/2, ct[0] + w/2, ct[1] + h/2, 1] +
                              pts[:, :2].reshape(num_joints*2).tolist() +
                              [cls_id])

        #if rot != 0:
        #    hm = hm * 0 + 0.9999
        #    reg_mask *= 0
        #    kps_mask *= 0
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
               'hps': kps, 'hps_mask': kps_mask}

        if self.opts.dense_hp:
            dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
            dense_kps_mask = dense_kps_mask.reshape(
                num_joints, 1, output_res, output_res)
            dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
            dense_kps_mask = dense_kps_mask.reshape(
                num_joints * 2, output_res, output_res)
            ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
            del ret['hps'], ret['hps_mask']
        if self.opts.reg_offset:
            ret.update({'reg': reg})
        if self.opts.hm_hp:
            ret.update({'hm_hp': hm_hp})
        if self.opts.reg_hp_offset:
            ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
        if self.opts.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 40), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret

