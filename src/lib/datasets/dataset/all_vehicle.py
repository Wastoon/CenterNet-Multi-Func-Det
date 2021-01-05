from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class ALLVehicle(data.Dataset):
  num_classes = 31
  default_resolution = [960, 960]
  #mean = np.array([0.40789654, 0.44719302, 0.47026115],
  #                 dtype=np.float32).reshape(1, 1, 3)
  #std  = np.array([0.28863828, 0.27408164, 0.27809835],
  #                 dtype=np.float32).reshape(1, 1, 3)
  mean = np.array([0.5194416012442385, 0.5378052387430711, 0.533462090585746], dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.3001546018824507, 0.28620901391179554, 0.3014112676161966], dtype=np.float32).reshape(1, 1, 3)


  def __init__(self, opt, split):
    super(ALLVehicle, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'Vehicle')
    self.img_dir = os.path.join(self.data_dir, 'all_images')
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          'val.json').format(split)
    if opt.use_contour_fourier:
      self.annot_path = os.path.join(
        self.data_dir, 'annotations',
        '{}_contour_area.json').format(split)
    else:
      self.annot_path = os.path.join(
        self.data_dir, 'annotations',
        'all_image_{}.json').format(split)


    self.max_objs = 128
    self.class_name = [
      '__background__', 'car_white', 'suv_red', 'suv_white', 'car_black', 'car_red', 'suv_black', 'van_white', 'suv_yellow',
     'car_yellow', 'car_blue', 'car_unknown', 'other', 'truck_white','van_black', 'bus_green' ,'truck_blue', 'bus_black',
      'bus_white', 'truck_red', 'car_green',  'truck_green','truck_black', 'truck_unknown', 'suv_unknown', 'van_yellow',
      'truck_yellow','van_unknown', 'van_green', 'van_blue', 'car_silver_gray', 'bus_yellow']
    self._valid_ids = [1, 2, 3, 4, 5, 6,7 ,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing Vehicle full {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)
    self.order = self.opt.fourier_order

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[5]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }

          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results),
                open('{}/results.json'.format(save_dir), 'w'))

  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))

    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__== '__main__':
  anapath = '/data4/mry/Vehicle/annotations/val.json'
  coco = coco.COCO(anapath)
  coco_dets = coco.loadRes('/data1/mry/code/centernet_newversion/exp/vehicle_det/vehicle_debug_dla34/results.json')
  coco_eval = COCOeval(coco, coco_dets, "bbox")
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()