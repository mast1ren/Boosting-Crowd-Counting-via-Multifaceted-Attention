from re import sub
from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse
import json


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(
        square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis


def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('.jpg', '.mat').replace('data', 'annotation')
    points = loadmat(mat_path)['locations'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * \
        (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='../../ds/dronebird',
                        help='original data directory')
    parser.add_argument('--data-dir', default='./preprocessed_data',
                        help='processed data directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 512
    max_size = 2048

    for phase in ['train', 'test']:
        sub_dir = os.path.join(args.origin_dir, phase)
        if phase == 'train':
            sub_phase_list = ['train', 'val']
            for sub_phase in sub_phase_list:
                sub_save_dir = os.path.join(save_dir, sub_phase)
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                with open(os.path.join(args.origin_dir, sub_phase + '.json'), 'r') as f:
                    paths = json.load(f)
                i = 0
                for path in paths:
                    im_path = path
                    name = im_path.split('/')[-3] + \
                        '_' + os.path.basename(im_path)
                    im, points = generate_data(im_path)
                    if sub_phase == 'train':
                        dis = find_dis(points)
                        points = np.concatenate([points, dis], axis=1)
                    im.save(os.path.join(sub_save_dir, name))
                    np.save(os.path.join(sub_save_dir,
                            name.replace('.jpg', '.npy')), points)
                    print('\r{}: {}/{}'.format(sub_phase, i, len(paths)), end='')
                    i += 1
                print()
        else:
            sub_save_dir = os.path.join(save_dir, 'test')
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            with open(os.path.join(args.origin_dir, 'test.json'), 'r') as f:
                paths = json.load(f)
            im_list = paths
            # im_list = glob(os.path.join(sub_dir, '*jpg'))
            i = 0
            for im_path in im_list:
                name = im_path.split('/')[-3]+'_'+os.path.basename(im_path)
                # print(name)
                im, points = generate_data(im_path)
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)
                print('\r{}: {}/{}'.format(phase, i, len(im_list)), end='')
                i += 1
            print()
