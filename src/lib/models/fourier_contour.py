import torch
import numpy as np
from math import tau
import cv2
from .utils import _gather_feat, _tranpose_and_gather_feat, _tranpose_and_gather_feat_gridneighbor
from .decode import _nms, _topk
###input : batch x K x 128 x 128

def fourier_contour_decode(fourier_contour_map, center_index, order=100, K=100):
    batch = fourier_contour_map.size(0)
    contour = _tranpose_and_gather_feat(fourier_contour_map, center_index) ## batch x (K xorders)
    contour = contour.view(batch, K, (2*order+1)*2)
    return contour

def de_scale_fourier_coef(contour_coef, scale):
    """
    contour_coef:  batch x K x (2*order+1)*2
    scale: scalar
    return : decoded_scale_coef: batch x K x (2*order+1)*2
    """
    return contour_coef * scale

def de_translation_fourier_coef(contour_coef, translation_x, translation_y):
    """
    contour_coef:  batch x K x (2*order+1)*2
    translation_x: scalar
    translation_y: scalar
    return : de_translation_fourier_coef: batch x K x (2*order+1)*2
    """
    order = contour_coef.shape[2] // 4
    contour_coef[:, :, 2*order : 2*order+2] += torch.tensor([translation_x, translation_y])
    return contour_coef


def DFT(t, coef_list, order=5):
    kernel = np.array([np.exp(-n*1j*t) for n in range(-order, order+1)])
    series = np.sum( (coef_list[:,0]+1j*coef_list[:,1]) * kernel[:])
    return np.real(series), np.imag(series)

def reproduce_shape_by_fourier(coef, order=100, time_length=300):
    """
    coef: 2*order+1 x 2
    return : time_length x 2(results of DFT)
    """
    space = np.linspace(0, tau, time_length)
    dft_results = np.zeros((time_length, 2))

    for idx, t in enumerate(space):
        real_DFT, imag_DFT = DFT(t, coef, order)
        dft_results[idx] = np.array([real_DFT, imag_DFT])
    return dft_results


def decode_batch_curve(fourier_contour_map, center_index, down_ratio, order=100, time_length=300, K=100):
    extract_contour = fourier_contour_decode(fourier_contour_map, center_index)

    de_scale_fourier_coef(extract_contour, down_ratio)
    de_translation_fourier_coef(extract_contour, 0.0, 0.0)


    batch = extract_contour.size(0)
    decoded_contour = np.zeros((batch, K, time_length, 2))

    for i in range(batch):
        for j in range(K):
            coef_list = extract_contour[i][j].reshape(-1, 2).numpy()
            assert coef_list.shape[0] == order
            decoded_peer_contour = reproduce_shape_by_fourier(coef_list, order=order, time_length=time_length)
            decoded_contour[i][j] = decoded_peer_contour

    return decoded_contour ## batch x K x time_length x 2 (960 x 960 scale)

def trans_curve2_obj_center(decoded_contour, center_xy_coordinate):
    """
    decoded_contour: batch x K x time_length x 2
    center_xy_coordinate: batch x K x 2
    """
    center_xy_coordinate[:,:,1] = center_xy_coordinate[:,:,1] * (-1)
    center_xy_coordinate = center_xy_coordinate.unsqueeze(2)
    decoded_contour = center_xy_coordinate + decoded_contour ##batch x K x time_length x 2
    decoded_contour[:,:,:,1] = decoded_contour[:,:,:,1] * (-1)
    return decoded_contour




