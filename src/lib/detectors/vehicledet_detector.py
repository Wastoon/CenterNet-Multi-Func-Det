from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import os

try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import vehicledet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import vehicledet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector


class VehicledetDetector(BaseDetector):
    def __init__(self, opt):
        super(VehicledetDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            ang = output['ang'].relu_()
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = vehicledet_decode(hm, wh, ang, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = vehicledet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                temp = np.concatenate([results[j][:,:4], results[j][:,-1:]], axis=1)
                soft_nms(temp, Nt=0.5, method=2)
                results[j][:,:4] = temp[:,:4]
                results[j][:,-1:] = temp[:,-1:]
        scores = np.hstack(
            [results[j][:, 5] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 5] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            self.vis_hm = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            #print(img.shape, pred.shape)
            debugger.add_blend_img(img, self.vis_hm, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='angled_pred_out')
            for k in range(len(dets[i])):
                if detection[i, k, 5] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 5],
                                           img_id='out_pred_{:.1f}'.format(scale),
                                           show_txt=self.opt.show_label)

                    debugger.add_coco_Rotate_bbox(detection[i, k, :4], detection[i, k, -1],
                                                  detection[i, k, 4], detection[i, k, 5], img_id='angled_pred_out',
                                                  show_txt=self.opt.show_label)

    def show_results(self, debugger, image, results):
        #print(image.shape)
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[5] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[5], img_id='ctdet', show_txt=self.opt.show_label)
        debugger.show_all_imgs(pause=self.pause)
        # prefix = image_name.split('.')[0]
        path = os.path.dirname(self.opt.det_output_path) + '/img'
        # debugger.save_all_imgs(path, prefix)

    def results_for_video(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[5] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[5], img_id='ctdet', show_txt=self.opt.show_label)
        #debugger.add_blend_img(image, self.vis_hm, 'ctdet')
        vis_img = debugger.show_onekind_imgs(img_id='ctdet', pause=False)
        return vis_img

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            yield self.run(frame)['vis_img']


    def save_results_only(self, debugger, image, results, image_name):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[5] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[5], img_id='ctdet')
        prefix = image_name.split('.')[0]
        path = os.path.dirname(self.opt.det_output_path) + '/img'
        debugger.save_all_imgs(path, prefix)

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        image_name = None
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
            import os
            image_name = os.path.basename(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
                # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            images = images.to(self.opt.device)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            output, dets, forward_time = self.process(images, return_time=True)
            #print(image_name)
            #print(dets)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            if self.opt.debug >= 2 and self.opt.debug <= 5:
                self.debug(debugger, images, dets, output, scale)

            dets = self.post_process(dets, meta, scale)
            # print(dets)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1 and self.opt.debug <= 5:
            if image_name is not None:
                self.show_results(debugger, image, results)
                # self.save_results_only(debugger, image, results, image_name)
                # self.save_person_only(debugger, image, results, image_name)

        if self.opt.debug == 5:
            vis_img = self.results_for_video(debugger, image, results)
        if self.opt.debug == 0:
            if image_name is not None:
                self.save_results_only(debugger, image, results, image_name)

        return_val = {'results': results, 'tot': tot_time, 'load': load_time,
                      'pre': pre_time, 'net': net_time, 'dec': dec_time,
                      'post': post_time, 'merge': merge_time, 'image_name': image_name}
        if self.opt.debug == 5:
            return_val['vis_img'] = vis_img

        return return_val


#python src/demo.py --gpus 3 --demo /data4/mry/Vehicle/ground_parking_pad_47/2.jpg  --load_model /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_angle_all_vehicle/model_60.pth  --task vehicle_det --dataset ALLVehicle --debug 2 --arch dla_34   --vis_thresh 0.3 --center_thresh 0.3 --show_label --test_scales '0.7'
#python src/demo.py --gpus 3 --demo /data4/mry/Vehicle/ground_parking_pad_47/13.jpg  --load_model /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_angle_all_vehicle/model_20.pth  --task vehicle_det --dataset ALLVehicle --debug 2 --arch dla_34  --test_scales '0.5,0.6,0.7,0.8,0.9,1,1.1,1.2' --vis_thresh 0.3 --nms --center_thresh 0.3
