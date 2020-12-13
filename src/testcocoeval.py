from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageDraw

class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        images, meta = {}, {}
        for scale in opt.test_scales:
            if opt.task == 'ddd':
                images[scale], meta[scale] = self.pre_process_func(
                    image, scale, img_info['calib'])
            else:
                images[scale], meta[scale] = self.pre_process_func(image, scale)
        return img_id, {'images': images, 'image': image, 'meta': meta}

    def __len__(self):
        return len(self.images)




def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)


    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)


    coco_dets = json.load(open('/data/mry/code/CenterNet/exp/landmark/hg_1x/results.json', 'r'))
    coco_gt = coco.COCO('/data/mry/DataSet/landmark/300W/paper_public_use/annotations/val.json')
    coco_eval = COCOeval(coco_gt, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()



if __name__ == '__main__':
    coco_gt = coco.COCO('/data/mry/DataSet/landmark/300W/paper_public_use/annotations/val.json')
    coco_dets_hmreg = coco_gt.loadRes('/home/mry/Desktop/results/hm_reg/common/results.json')
    coco_dets_reg = coco_gt.loadRes('/home/mry/Desktop/results/reg/common/results.json')

    gt_np = np.zeros((68, 2))
    dt_np = np.zeros((68, 2))
    dt_reg= np.zeros((68, 2))
    for i in range(200):
        img_id = i

        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))[0]
        dts = coco_dets_hmreg.loadAnns(coco_dets_hmreg.getAnnIds(imgIds=img_id))[0]
        dts_reg = coco_dets_reg.loadAnns(coco_dets_hmreg.getAnnIds(imgIds=img_id))[0]
        for i in range(68):
            dt_np[i] = [dts['keypoints'][3 * i], dts['keypoints'][3 * i + 1]]
        for i in range(68):
            gt_np[i] = [gts['keypoints'][3 * i], gts['keypoints'][3 * i + 1]]
        for i in range(68):
            dt_reg[i] = [dts_reg['keypoints'][3 * i], dts_reg['keypoints'][3 * i + 1]]
    print(gt_np[1], dt_np[1], dt_reg[1])


    img = np.zeros((1000,1000,3), np.uint8)
    point_size = 1
    point_color_dt = (0, 0,255)
    point_color_gt = (0, 255,0)
    point_color_dt_reg = (255, 255, 255)
    for idx, i in enumerate(dt_np) :
        cv2.circle(img, (int(i[0]), int(i[1])), point_size, point_color_dt, thickness=2)
    for idx, i in enumerate(dt_reg) :
        cv2.circle(img, (int(i[0]), int(i[1])), point_size, point_color_dt_reg, thickness=2)
    for i in gt_np:
        cv2.circle(img, (int(i[0]), int(i[1])), point_size, point_color_gt, thickness=2)

    cv2.namedWindow("image")
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


