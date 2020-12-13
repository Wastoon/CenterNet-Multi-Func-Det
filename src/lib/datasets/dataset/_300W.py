import os
import cv2
import time
import torch.utils.data as data
import numpy as np
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import json

class _300W(data.Dataset):
    #person only
    num_classes = 1
    #person joints numbers in coco
    num_joints = 68
    #input default resolution is 256x256
    default_resolution = [256, 256]
    ##for data augment
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    #filp index
    flip_idx = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], [17, 26], [18, 25],
                [19, 24], [20, 23], [21, 22], [39, 42], [38, 43], [37, 44], [36, 45], [41, 46], [40, 47],
                [31, 35], [32, 34], [48, 54], [49, 53], [50, 52], [59, 55], [58, 56], [60, 64], [61, 63],
                [67, 65]]

    def __init__(self, opts, split, test_split=None):
        super(_300W, self).__init__()
        #person joints in coco connection or not==whether exists line between 2 joints
        self.edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
                      [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20], [20, 21],
                      [22, 23], [23, 24], [24, 25], [25, 26], [27, 28], [28, 29], [29, 30], [31, 32], [32, 33],
                      [33, 34], [34, 35], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41], [42, 43],
                      [43, 44], [44, 45], [45, 46], [46, 47], [42, 47], [48, 49], [49, 50], [50, 51], [51, 52],
                      [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [48, 59], [60, 61],
                      [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [60, 67]]
        self.test_split = test_split
        if test_split:
            # data dir
            self.data_dtr = os.path.join(opts.data_dir, 'paper_public_use')
            # image_dir
            self.img_dir = os.path.join(self.data_dtr, 'images', '{}'.format(test_split))
            # annotation_dir
            self.annotation_dir = os.path.join(self.data_dtr, 'annotations', '{}.json'.format(test_split))
            # max number of person COCO_kp included in one img
            self.max_objs = 32
            # setting random seed
            self.data_rng = np.random.RandomState(1997)
            # split method
            self.split = split
            # options for program
            self.opts = opts
            # for data color augment
            self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                     dtype=np.float32)
            self._eig_vec = np.array([
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938]
            ], dtype=np.float32)

            print('===========>>>>>>initializing the 300W_{} data, Begin loading'.format(test_split))
            self.coco = coco.COCO(self.annotation_dir)
            # image_ids is a list of img's id
            image_ids = self.coco.getImgIds()

            if split == 'train':
                self.images = []
                for img_id in image_ids:
                    idxs = self.coco.getAnnIds(imgIds=[img_id])
                    if len(idxs) > 0:
                        self.images.append(img_id)
            else:
                self.images = image_ids
            # num of img to train or val
            self.num_samples = len(self.images)
            print('Loaded {} {} samples'.format(test_split, self.num_samples))

        else:
            # data dir
            self.data_dtr = os.path.join(opts.data_dir, 'paper_public_use')
            # image_dir
            self.img_dir = os.path.join(self.data_dtr, 'images', '{}'.format(split))
            # annotation_dir
            self.annotation_dir = os.path.join(self.data_dtr, 'annotations', '{}.json'.format(split))
            # max number of person COCO_kp included in one img
            self.max_objs = 32
            # setting random seed
            self.data_rng = np.random.RandomState(1997)
            # split method
            self.split = split
            # options for program
            self.opts = opts
            # for data color augment
            self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                     dtype=np.float32)
            self._eig_vec = np.array([
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938]
            ], dtype=np.float32)

            print('===========>>>>>>initializing the 300W_{} data, Begin loading'.format(split))
            self.coco = coco.COCO(self.annotation_dir)
            # image_ids is a list of img's id
            image_ids = self.coco.getImgIds()

            if split == 'train':
                self.images = []
                for img_id in image_ids:
                    idxs = self.coco.getAnnIds(imgIds=[img_id])
                    if len(idxs) > 0:
                        self.images.append(img_id)
            else:
                self.images = image_ids
            # num of img to train or val
            self.num_samples = len(self.images)
            print('Loaded {} {} samples'.format(test_split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = 1
                for dets in all_bboxes[image_id][cls_ind]:
                    bbox = dets[:4]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = dets[4]
                    bbox_out = list(map(self._to_float, bbox))
                    keypoints = np.concatenate([
                        np.array(dets[5:141], dtype=np.float32).reshape(-1, 2),
                        np.ones((68, 1), dtype=np.float32)], axis=1).reshape(204).tolist()
                    keypoints = list(map(self._to_float, keypoints))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score)),
                        "keypoints": keypoints
                    }
                    if score > 0.3:
                        detections.append(detection)
                        break
                break
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(opt.save_dir, "results.json")
        # detections  = convert_eval_format(all_boxes)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def _300w_run_eval(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))
        coco_gt = coco.COCO(self.annotation_dir)
        coco_dets = coco_gt.loadRes('{}/results.json'.format(save_dir))

        miss_list = []

        total_nme_ipn = 0
        total_count = 0
        fail_count_ipn = 0
        total_nme_ion = 0
        fail_count_ion = 0
        valid_total_num = 0
        miss_landmark = 0
        for img_id in range(self.num_samples):
            gt_np = np.zeros((68, 2))
            dt_np = np.zeros((68, 2))

            anno_gt_idx = coco_gt.getAnnIds(imgIds=img_id)
            anno_dt_idx = coco_dets.getAnnIds(imgIds=img_id)

            if bool(anno_dt_idx):
                valid_total_num += 1
                gts = coco_gt.loadAnns(anno_gt_idx)[0]
                dts = coco_dets.loadAnns(anno_dt_idx)[0]
                for i in range(68):
                    dt_np[i] = [dts['keypoints'][3 * i], dts['keypoints'][3 * i + 1]]
                    gt_np[i] = [gts['keypoints'][3 * i], gts['keypoints'][3 * i + 1]]

                #####IPN
                left_eye = np.average(gt_np[36:42], axis=0)
                right_eye = np.average(gt_np[42:48], axis=0)
                norm_factor_ipn = np.linalg.norm(left_eye - right_eye)

                ####ION
                left_corner = gt_np[36]
                right_corner = gt_np[45]
                norm_factor_ion = np.linalg.norm(left_corner - right_corner)

                single_nme_ipn = (np.sum(np.linalg.norm(dt_np-gt_np, axis=1)) / dt_np.shape[0]) / norm_factor_ipn
                single_nme_ion = (np.sum(np.linalg.norm(dt_np - gt_np, axis=1)) / dt_np.shape[0]) / norm_factor_ion
                total_nme_ipn += single_nme_ipn
                total_nme_ion += single_nme_ion
                total_count += 1
                if single_nme_ipn > 0.1:
                    fail_count_ipn += 1
                if single_nme_ion > 0.1:
                    fail_count_ion += 1

            else:
                miss_landmark += 1
                img_idx = coco_gt.getImgIds(imgIds=img_id)
                img_name = coco_gt.loadImgs(ids=img_idx)[0]['file_name']
                img_path = os.path.join(self.img_dir, img_name)
                img = cv2.imread(img_path)
                height, width = img.shape[0], img.shape[1]
                miss_list.append((img_path, width, height))
        epoch_nme_ipn = total_nme_ipn / valid_total_num
        epoch_nme_ion = total_nme_ion / valid_total_num
        print('Ipn metric:NME: {:.6f} Failure Rate: {:.6f} Total Count: {:.6f} Fail Count: {:.6f}'.format(epoch_nme_ipn,
                                                                                              fail_count_ipn / total_count,
                                                                                             total_count, fail_count_ipn))

        print('Ion metric:NME: {:.6f} Failure Rate: {:.6f} Total Count: {:.6f} Fail Count: {:.6f}'.format(epoch_nme_ion,
                                                                                               fail_count_ion / total_count,
                                                                                               total_count, fail_count_ion))

        miss_landmark_rate = miss_landmark / self.num_samples
        print('Miss Landmark:Miss Rate: {:.6f} Dataset Num: {:.6f} Miss Num: {:.6f}'.format(miss_landmark_rate,
                                                                                           self.num_samples,miss_landmark))

        fw = open(os.path.join(save_dir, '{}_miss_imgs.txt'.format(self.test_split)), 'w')
        for line in miss_list:
            fw.writelines(line[0] +' ' + 'solution:{}x{}'.format(line[1], line[2])+ '\n')
        fw.close()

        #iStr = ' {:<10} {} [ NME={:6f} | failure_rate={:6f} | Total_count={:<4} | failure_count={:<5} | fr={:<4}={}/{}]'
        #title1 = 'Ipn_metric'
        #title2 = 'Ion_metric'
        #dataset = self.test_split
        #str2 = iStr.format(title2, dataset, epoch_nme_ion, fail_count_ion / total_count, total_count, fail_count_ion, miss_landmark_rate, miss_landmark, self.num_samples)
        #str1 = iStr.format(title1, dataset, epoch_nme_ipn, fail_count_ipn / total_count, total_count, fail_count_ipn, miss_landmark_rate, miss_landmark, self.num_samples)
        #print(str2)
        #print(str1)