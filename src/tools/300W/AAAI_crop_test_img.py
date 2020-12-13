# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PIL import Image, ImageDraw
import os
import numpy as np
from matplotlib.pyplot import imshow
# import cv2
import math
import json


def load_keypoint_from_pts(img_pts):
    fr = open(img_pts, 'r').readlines()
    valid_info = fr[3:-1]
    point_dict = {}
    for idx, pixel_coordinate in enumerate(valid_info):
        point_x, point_y = (float)(pixel_coordinate.strip().split(' ')[0]), (float)(
            pixel_coordinate.strip().split(' ')[1])
        point_dict[int(idx)] = (point_x, point_y)
    return point_dict


def load_img(img_path):
    img = Image.open(img_path)
    # img.show()
    return img


def show_keypoint(img, keypoint_dict):
    draw = ImageDraw.Draw(img)
    for kepoint in keypoint_dict.keys():
        draw.point(keypoint_dict[kepoint])
    img.show()


def crop_head(img, keypoint_dict, wanna_size):
    points_coordinate = np.zeros([len(keypoint_dict), 2])
    for key_point in keypoint_dict.keys():
        points_coordinate[key_point] = keypoint_dict[key_point]
    min_left = np.min(points_coordinate, axis=0)[0]
    max_right = np.max(points_coordinate, axis=0)[0]
    center_x = (max_right - min_left) / 2 + min_left
    print(center_x)
    crop_box_left, crop_box_right = center_x - wanna_size / 2, center_x + wanna_size / 2

    lefteye_center = np.mean(points_coordinate[36:42], axis=0)
    righteye_center = np.mean(points_coordinate[42:48], axis=0)
    print(lefteye_center)

    eye_center = (lefteye_center[1] + righteye_center[1]) / 2

    lip_center = np.mean(points_coordinate[48:68], axis=0)
    print(lip_center, eye_center)
    middle_length = lip_center[1] - eye_center

    crop_box_top, crop_box_bottom = eye_center - (wanna_size - middle_length) / 2, lip_center[1] + (
                wanna_size - middle_length) / 2

    crop_box_left, crop_box_top, crop_box_right, crop_box_bottom = [int(i) for i in
                                                                    [crop_box_left, crop_box_top, crop_box_right,
                                                                     crop_box_bottom]]
    crop_img = img.crop((crop_box_left, crop_box_top, crop_box_right, crop_box_bottom))
    crop_img.show()
    return crop_box_left, crop_box_top, crop_box_right, crop_box_bottom, crop_img


def crop_head_v2(img, keypoint_dict, wanna_size):
    points_coordinate = np.zeros([len(keypoint_dict), 2])
    for key_point in keypoint_dict.keys():
        points_coordinate[key_point] = keypoint_dict[key_point]
    max_left = np.min(points_coordinate, axis=0)[0]
    max_right = np.max(points_coordinate, axis=0)[0]
    max_top = np.min(points_coordinate, axis=0)[1]
    max_bottom = np.max(points_coordinate, axis=0)[1]

    fake_left = abs(points_coordinate[34][0] - max_left)
    fake_right = abs(points_coordinate[34][0] - max_right)
    fake_top = abs(points_coordinate[34][1] - max_top)
    fake_bottom = abs(points_coordinate[34][1] - max_bottom)
    max_radical = max(fake_bottom, fake_left, fake_right, fake_top)
    center_x, center_y = points_coordinate[34][0], points_coordinate[34][1]
    if max_radical < wanna_size / 2:
        max_radical = wanna_size / 2
        left, right = center_x - max_radical, center_x + max_radical
        top, bottom = center_y - max_radical, center_y + max_radical
    else:
        expand_factor = 10
        left, right = center_x - max_radical - expand_factor, center_x + max_radical + expand_factor
        top, bottom = center_y - max_radical - expand_factor, center_y + max_radical + expand_factor
    crop_box_left, crop_box_top, crop_box_right, crop_box_bottom = [int(i) for i in [left, top, right, bottom]]
    crop_img = img.crop((crop_box_left, crop_box_top, crop_box_right, crop_box_bottom))
    print(crop_img.size)
    # crop_img.show()

    return crop_box_left, crop_box_top, crop_box_right, crop_box_bottom, crop_img


def calculate_coordinate_crop(keypoint_dict, crop_box_top, crop_box_left):
    for key_point in keypoint_dict.keys():
        keypoint_dict[key_point] = (
        keypoint_dict[key_point][0] - crop_box_left, keypoint_dict[key_point][1] - crop_box_top)
    return keypoint_dict


def process_categories():
    category_dict = {'supercategory': 'person',
                     'keypoints': [i for i in range(1, 69)],
                     'skeleton': [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
                                  [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17],
                                  [18, 19], [19, 20], [20, 21], [21, 22], [23, 24], [24, 25], [25, 26], [26, 27],
                                  [28, 29], [29, 30], [30, 31],
                                  [32, 33], [33, 34], [34, 35], [35, 36],
                                  [37, 38], [38, 39], [39, 40], [40, 41], [41, 42], [37, 42],
                                  [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [43, 48],
                                  [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57],
                                  [57, 58], [58, 59], [59, 60], [49, 60],
                                  [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 68], [61, 68]],

                     'id': 1,
                     'name': 'person'}
    return category_dict


def _300w2cocokp(org_img_dir, tar_img_dir, org_video_dir=None):
    anno_dict = {'license': [],
                 'categories': [],
                 'images': [],
                 'annotations': [],
                 'info': {}}

    img_list = []
    pts_list = []
    for adir in org_img_dir:
        files = os.listdir(adir)

        for file in files:
            file_path = os.path.join(adir, file)
            if file.split('.')[1] == 'pts':
                pts_list.append(file_path)
            else:
                img_list.append(file_path)
    img_list.sort()
    pts_list.sort()


    if org_video_dir is not None:
        for video_dir in org_video_dir:
            video_img_list = []
            video_pts_list = []
            img_base_path = os.path.join(video_dir, 'extraction')
            files = os.listdir(img_base_path)
            for file in files:
                file_path = os.path.join(img_base_path, file)
                video_img_list.append(file_path)

            pts_base_path = os.path.join(video_dir, 'annot')
            files = os.listdir(pts_base_path)
            for file in files:
                file_path = os.path.join(pts_base_path, file)
                video_pts_list.append(file_path)
            video_img_list.sort()
            video_pts_list.sort()
            for idx in range(len(video_img_list)) :
                img_pts = video_pts_list[idx]
                point_dict = load_keypoint_from_pts(img_pts)
                img_path = video_img_list[idx]
                org_img = load_img(img_path)
                crop_box_left, crop_box_top, crop_box_right, crop_box_bottom, crop_img = crop_head_v2(org_img,
                                                                                                      point_dict, 224)

                new_file_name = str(idx).zfill(6) + '.jpg'
                save_dir = os.path.join(tar_img_dir, os.path.basename(video_dir))
                checkdir(save_dir)
                new_file_path = os.path.join(save_dir, new_file_name)

                crop_img.save(new_file_path, "JPEG")


    image_list = []
    annotations_list = []
    for idx in range(len(img_list)):
        img_pts = pts_list[idx]
        point_dict = load_keypoint_from_pts(img_pts)
        img_path = img_list[idx]
        org_img = load_img(img_path)
        crop_box_left, crop_box_top, crop_box_right, crop_box_bottom, crop_img = crop_head_v2(org_img, point_dict, 224)
        new_file_name = str(idx).zfill(6) + '.jpg'
        save_dir = os.path.join(tar_img_dir, '300W')
        checkdir(save_dir)
        new_file_path = os.path.join(save_dir ,new_file_name)
        crop_img.save(new_file_path, "JPEG")


    return anno_dict


def checkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':

    img_train_dir = '/data1/mry/datasets/landmark-datasets/300W/paper_public_use_test/images/train'
    img_test_dir = '/data1/mry/datasets/landmark-datasets/300W/paper_public_use_test/images/val'
    img_common_set = '/data1/mry/datasets/landmark-datasets/300W/paper_public_use_test/images/commonsubset'
    img_challenge_set = '/data1/mry/datasets/landmark-datasets/300W/paper_public_use_test/images/challengesubset'
    img_full_set = '/data1/mry/datasets/landmark-datasets/300W/paper_public_use_test/images/fullset'
    anno_dir = '/data1/mry/datasets/landmark-datasets/300W/paper_public_use_test/annotations'

    img_AAAI_train_dir = '/data1/mry/datasets/landmark-datasets/300W/paper_public_use_test/images/AAAI_train'
    img_AAAI_test_dir = '/data1/mry/datasets/landmark-datasets/300W/paper_public_use_test/images/AAAI_val'
    if not os.path.exists(img_train_dir):
        os.makedirs(img_train_dir)
    if not os.path.exists(img_test_dir):
        os.makedirs(img_test_dir)
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)
    checkdir(img_common_set)
    checkdir(img_challenge_set)
    checkdir(img_full_set)

    train_img_dir = ['/data1/mry/datasets/landmark-datasets/300W/lfpw/trainset',
                     '/data1/mry/datasets/landmark-datasets/300W/helen/trainset', '/data1/mry/datasets/landmark-datasets/300W/afw']
    test_img_dir = ['/data1/mry/datasets/landmark-datasets/300W/helen/testset',
                    '/data1/mry/datasets/landmark-datasets/300W/lfpw/testset', '/data1/mry/datasets/landmark-datasets/300W/ibug']
    full_img_dir = ['/data1/mry/datasets/landmark-datasets/300W/helen/testset',
                    '/data1/mry/datasets/landmark-datasets/300W/lfpw/testset', '/data1/mry/datasets/landmark-datasets/300W/ibug']

    common_img_dir = ['/data1/mry/datasets/landmark-datasets/300W/helen/testset',
                    '/data1/mry/datasets/landmark-datasets/300W/lfpw/testset']

    challenge_img_dir = ['/data1/mry/datasets/landmark-datasets/300W/ibug']

    AAAI_train_img_dir = ['/data1/mry/datasets/landmark-datasets/300W/lfpw/trainset',
                          '/data1/mry/datasets/landmark-datasets/300W/helen/trainset',
                          '/data1/mry/datasets/landmark-datasets/300W/afw',
                          '/data1/mry/datasets/landmark-datasets/300W/helen/testset',
                          '/data1/mry/datasets/landmark-datasets/300W/lfpw/testset',
                          '/data1/mry/datasets/landmark-datasets/300W/ibug']
    AAAI_test_img_dir =  ['/data1/mry/datasets/landmark-datasets/300W/helen/testset',
                    '/data1/mry/datasets/landmark-datasets/300W/lfpw/testset', '/data1/mry/datasets/landmark-datasets/300W/ibug']

    AAAI_full_img_dir = ['/data1/mry/datasets/landmark-datasets/300W/helen/testset',
                    '/data1/mry/datasets/landmark-datasets/300W/lfpw/testset', '/data1/mry/datasets/landmark-datasets/300W/ibug']

    AAAI_common_img_dir = ['/data1/mry/datasets/landmark-datasets/300W/helen/testset',
                    '/data1/mry/datasets/landmark-datasets/300W/lfpw/testset']

    AAAI_challenge_img_dir = ['/data1/mry/datasets/landmark-datasets/300W/ibug']

    AAAI_test_video_num = [410, 411, 516, 517, 526, 528, 529, 530, 531, 533, 557, 558, 559, 562]
    AAAI_train_video_num = [ '002', '033', '020',  '119',  '025', '205', '047', '007', '013',
                 '028', '225', '041',  '160', '001', '138', '057', '044', '037', '019', '022']

    video_base_dir = '/data1/mry/datasets/landmark-datasets/300VW_Dataset_2015_12_14/'
    AAAI_test_video_img_dir = [os.path.join(video_base_dir, str(i)) for i in AAAI_test_video_num]
    AAAI_train_video_img_dir = [os.path.join(video_base_dir, i) for i in AAAI_train_video_num]
    AAAI_video_dir = AAAI_train_video_img_dir + AAAI_test_video_img_dir

    fw = open(os.path.join(anno_dir, 'train.json'), 'w')
    train_anno_dict = _300w2cocokp(AAAI_train_img_dir, img_train_dir, AAAI_video_dir)
    json.dump(train_anno_dict, fw)


    fw = open(os.path.join(anno_dir, 'val.json'), 'w')
    test_anno_dict = _300w2cocokp(AAAI_test_img_dir, img_test_dir)
    json.dump(test_anno_dict, fw)


    fw = open(os.path.join(anno_dir, 'common.json'), 'w')
    common_anno_dict = _300w2cocokp(AAAI_common_img_dir, img_common_set)
    json.dump(common_anno_dict, fw)

    fw = open(os.path.join(anno_dir, 'challenge.json'), 'w')
    challenge_anno_dict = _300w2cocokp(AAAI_challenge_img_dir, img_challenge_set)
    json.dump(challenge_anno_dict, fw)

    fw = open(os.path.join(anno_dir, 'full.json'), 'w')
    full_anno_dict = _300w2cocokp(AAAI_full_img_dir, img_full_set)
    json.dump(full_anno_dict, fw)

    #fw = open(os.path.join(anno_dir, 'train.json'), 'w')
    #train_anno_dict = _300w2cocokp(train_img_dir, img_train_dir)
    #json.dump(train_anno_dict, fw)

    fw.close()




