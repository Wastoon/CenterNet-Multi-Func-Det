
import cv2
import os
import numpy as np

def scale_img(img, scale):
    height, width = img.shape[0], img.shape[1]
    new_height = int(scale * height)
    new_width  = int(scale * width)
    newimg = cv2.resize(img, (new_width, new_height))
    return newimg

def scale_img_xy(img, scalex, scaley):
    height, width = img.shape[0], img.shape[1]
    new_height = int(scaley * height)
    new_width  = int(scalex * width)
    newimg = cv2.resize(img, (new_width, new_height))
    return newimg

def rorate_img(img, rot_angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated
def save_scale_img(scale_img, save_path):
    cv2.imwrite(save_path, scale_img)

def checkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__=='__main__':
    img_path = '/data/mry/DataSet/300w/images/val/indoor_176.jpg'
    aug_path = '/data/mry/DataSet/300w/images/Aug'
    #miss_list_path = '/data/mry/code/CenterNet/exp/landmark/rot_aug/challenge_miss_imgs.txt'
    #fr = open(miss_list_path, 'r').readlines()
    #img_list = os.listdir('/data/mry/color')

    #for line in fr:
    #   img_path = line.split(' ')[0]

    file_name = os.path.basename(img_path).split('.')[0]
    #if file_name == '000045':
    #    print('find')
    dir_path = os.path.join(aug_path, file_name)
    checkdir(dir_path)
    for scale in np.arange(1,8.0,0.1):
        img = cv2.imread(img_path)
        img = rorate_img(img, 0)
        new_img = scale_img_xy(img, scale, scale*1)
        # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
        save_path = os.path.join(dir_path, str(round(scale, 2))+'_' + file_name + '.jpg')
        save_scale_img(new_img, save_path)
