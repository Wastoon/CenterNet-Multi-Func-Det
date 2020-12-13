
from .base_detector import BaseDetector
import torch
import time
from models.utils import clothlandmark_flip_lr_off, flip_tensor, flip_lr
from models.decode import deepfashion2_pose_decode
from utils.post_process import clothlandmark_post_process
import numpy as np
import os



try:
  from external.nms import soft_nms_39, soft_nms, soft_nms_593
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')

class ClothDetector(BaseDetector):
    def __init__(self, opt):
        super(ClothDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx

    def process(self, images, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()

            reg = output['reg'] if self.opt.reg_offset else None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()

            if self.opt.flip_test:
                output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
                output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
                output['hps'] = (output['hps'][0:1] +
                                 clothlandmark_flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
                hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                    if hm_hp is not None else None
                reg = reg[0:1] if reg is not None else None
                hp_offset = hp_offset[0:1] if hp_offset is not None else None

            dets = deepfashion2_pose_decode(
                output['hm'], output['wh'], output['hps'],
                reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = clothlandmark_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 593)
            # import pdb; pdb.set_trace()
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms_593(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])

        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results


    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:593] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())

        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if dets[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))


        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

            part_perd = debugger.gen_colormap_part_hp_debug(
                output['hm_hp'][0].detach().cpu().numpy(),  part_num=10
            )
            debugger.add_blend_img(img, part_perd, 'part_pred_hmhp')

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='deepfashion2landmark')

        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='deepfashion2landmark')
                    debugger.add_deepfashion2_hp(bbox[5:593], j -1, img_id='deepfashion2landmark')
        debugger.show_all_imgs(pause=self.pause)
        self.save_debug_img(debugger)



    def save_person_only(self, debugger, image, results, image_name):
        debugger.add_img(image, img_id='landmark')
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='landmark')
                debugger.add_coco_hp(bbox[5:593], img_id='landmark')
        imgId = image_name.split('.')[0]
        path =  self.opt.debug_dir
        if not os.path.exists(path):
            os.makedirs(path)
        debugger.save_person_only(imgId, pause=self.pause, path=path)

    def save_debug_img(self, debugger):
        debugger.save_debug_img(imgId='landmark', path=self.opt.debug_dir, pause=self.pause)
        debugger.save_debug_img(imgId='pred_hmhp', path=self.opt.debug_dir,
                                pause=self.pause)