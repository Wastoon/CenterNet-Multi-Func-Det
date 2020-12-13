import os
import cv2
import numpy as np
from PIL import Image

def checkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def split_img(bigimg, cell_h, cell_w):
    big_h, big_w = bigimg.shape[0], bigimg.shape[1]
    row = big_h // cell_h
    col = big_w // cell_w
    img_cells_list = []
    img_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    img_cell_w = np.zeros((cell_h, big_w - (col - 1) * cell_w, 3), dtype=np.uint8)
    img_cell_h = np.zeros((big_h - (row - 1) * cell_h, cell_w, 3), dtype=np.uint8)
    img_cell_wh = np.zeros((big_h - (row - 1) * cell_h, big_w - (col - 1) * cell_w, 3), dtype=np.uint8)
    flag = False
    flag_h = False
    flag_w = False
    flag_wh =False
    count = 0
    for i in range(col):
        for j in range(row):
            if i<col-1:
                for x in range(cell_w):
                    if j < row-1:
                        flag = True
                        for y in range(cell_h):
                            img_cell[y, x, :] = bigimg[ j * cell_h + y, i * cell_w + x, :]
                            #print(i,j)
                    else:
                        flag_h = True
                        for y in range(big_h-j*cell_h):
                            #print(y)
                            img_cell_h[y, x, :] = bigimg[ j * cell_h + y, i * cell_w + x, :]
                if flag:

                    img_cells_list.append(img_cell)
                    flag = False
                if flag_h:

                    img_cells_list.append(img_cell_h)
                    flag_h = False
            else:
                for x in range(big_w-i*cell_w):
                    if j < row -1:
                        flag_w = True
                        for y in range(cell_h):
                            img_cell_w[y, x, :] = bigimg[ j * cell_h + y, i * cell_w + x, :]
                    else:
                        flag_wh = True
                        for y in range(big_h-j*cell_h):
                            img_cell_wh[y, x, :] = bigimg[ j * cell_h + y, i * cell_w + x, :]
                if flag_w:
                    img_cells_list.append(img_cell_w)
                    flag_w = False
                if flag_wh:
                    img_cells_list.append(img_cell_wh)
                    flag_wh = False

    return img_cells_list


def split_imgv2(bigimg, cell_h, cell_w):
    big_h, big_w = bigimg.size[1], bigimg.size[0]
    row = big_h // cell_h
    col = big_w // cell_w
    step_h = cell_h
    step_w = cell_w
    img_cells_list = []
    img_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    img_cell_w = np.zeros((cell_h, big_w - (col - 1) * cell_w, 3), dtype=np.uint8)
    img_cell_h = np.zeros((big_h - (row - 1) * cell_h, cell_w, 3), dtype=np.uint8)
    img_cell_wh = np.zeros((big_h - (row - 1) * cell_h, big_w - (col - 1) * cell_w, 3), dtype=np.uint8)
    flag = -1
    flag_h = False
    flag_w = False
    flag_wh =False
    count = 0
    for i in range(col):
        for j in range(row):
            if i < col-1 and j < row-1:
                cell_h = step_h
                cell_w = step_w
                flag =0
            if i >= col-1 and j < row-1:
                cell_h = step_h
                cell_w = big_w-i*step_w
                flag = 1
            if i < col-1 and j >= row-1:
                cell_h = big_h-j*step_h
                cell_w = step_w
                flag = 2
            if i >= col-1 and j >=row-1:
                cell_h = big_h-j*step_h
                cell_w = big_w-i*step_w
                flag = 3

            cell = bigimg.crop((i * step_w, j * step_h, i * step_w+cell_w, j * step_h+cell_h))
            img_cells_list.append(cell)
            flag = -1

    return img_cells_list


if __name__=='__main__':
    img_path = '/data/mry/DataSet/300w/images/Aug/indoor_176/3.0_indoor_176.jpg'
    aug_path = '/data/mry/DataSet/300w/images/Aug'
    bigimg = Image.open(img_path)
    cell_h = 1000
    cell_w = 1600
    img_cells_list = split_imgv2(bigimg, cell_h, cell_w)
    file_name = os.path.basename(img_path).split('.')[0]
    dir_path = os.path.join(aug_path, file_name)
    checkdir(dir_path)
    for i, img in enumerate(img_cells_list):
        save_path = os.path.join(dir_path, str(i) + 'part_' + file_name + '.jpg')
        img.save(save_path)