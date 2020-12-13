from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data


class DEEPFASHION2(data.Dataset):
    num_classes = 13
    num_joints = 294
    default_resolution = [256, 256]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    flip_idx = [[1, 5], [2, 4], [6, 24], [7, 23], [8, 22], [9, 21], [10, 20], [11, 19], [12, 18], [13, 17], [14, 16],
                [26, 30], [27, 29], [31, 57], [32, 56], [33, 55], [34, 54], [35, 53], [36, 52], [37, 51], [38, 50], [39, 49],
                [40, 48], [41, 47], [42, 46], [43, 45], [61, 63], [60, 62], [59, 83], [64, 82], [65, 81], [66, 80], [67, 79],
                [68, 78], [69, 77], [70, 76], [71, 75], [72, 74], [73, 86], [88, 85], [87, 84], [90, 94], [91, 93], [92, 119],
                [95, 121], [96, 120], [97, 119], [98, 118], [99, 117], [100, 116], [101, 115], [102, 114], [103, 113], [104, 112],
                [105, 111], [106, 110], [107, 109], [108, 125], [127, 124], [126, 123], [129, 133], [130, 132], [134, 142],
                [135, 141], [136, 140], [137, 139], [149, 157], [144, 148], [145, 147], [150, 156], [151, 155], [152, 154],
                [158, 160], [161, 167], [162, 166], [163, 165], [168, 170], [171, 181], [172, 180], [173, 179], [174, 178],
                [175, 177], [182, 184], [185, 189], [186, 188], [191, 195], [192, 194], [196, 218], [197, 217], [198, 216],
                [199, 215], [200, 214], [201, 213], [202, 212], [203, 211], [204, 210], [205, 209], [206, 208], [220, 224],
                [221, 223], [225, 255], [226, 254], [227, 253], [228, 252], [229, 251], [230, 250], [231, 249], [232, 248],
                [233, 247], [234, 246], [235, 245], [236, 244], [237, 243], [238, 242], [239, 241], [257, 261], [258, 260],
                [262, 274], [263, 273], [264, 272], [265, 271], [266, 270], [267, 269], [281, 293], [276, 280], [277, 279],
                [282, 292], [283, 291], [284, 290], [285, 289], [286, 288]]


    def __init__(self, opt, split):
        super(DEEPFASHION2, self).__init__()
        self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [4, 6], [3, 5], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [6, 12], [5, 11], [11, 12],
                      [12, 14], [14, 16], [11, 13], [13, 15]]

        self.acc_idxs = [i for i in range(1, 295)]
        self.data_dir = os.path.join(opt.data_dir, 'deepfashion2_coco')
        self.img_dir = os.path.join(self.data_dir, 'images','{}'.format(split))
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                '{}imgpart4.json').format(split)
        else:
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'deepfashion_{}.json').format(split)
        self.max_objs = 15
        self.class_name=['__background__','short-sleeve-short', 'long-sleeve-short', 'short-sleeve-outwear', 'long-sleeve-outwear',
                          'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short-sleeve-dress', 'long-sleeve-dress',
                          'vest-dress', 'sling-dress']
        self._valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt

        print('==> initializing deepfashion2 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()

        if split == 'train':
            self.images = []
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if len(idxs) > 0:
                    self.images.append(img_id)
        else:
            self.images = image_ids
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for dets in all_bboxes[image_id][cls_ind]:
                    bbox = dets[:4]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = dets[4]
                    bbox_out = list(map(self._to_float, bbox))
                    keypoints = np.concatenate([
                        np.array(dets[5:593], dtype=np.float32).reshape(-1, 2),
                        np.ones((294, 1), dtype=np.float32)], axis=1).reshape(882).tolist()
                    keypoints = list(map(self._to_float, keypoints))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score)),
                        "keypoints": keypoints
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/S1results_part4.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(opt.save_dir, "results.json")
        # detections  = convert_eval_format(all_boxes)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/S1results_part4.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()