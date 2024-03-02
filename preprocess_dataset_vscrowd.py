from re import sub
from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse
import json
import random


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point * points, axis=1)
    dis = np.sqrt(
        np.maximum(
            square[:, None] - 2 * np.matmul(point, point.T) + square[None, :], 0.0
        )
    )
    # print(dis.shape)
    if dis.shape[0] > 3:
        dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    else:
        dis = np.mean(np.sort(dis,axis=1)[:, 1:], axis=1, keepdims=True)
    return dis


def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    print(im_path)
    file_path = os.path.basename(im_path).split('.')[0]
    # print(file_path)
    txt_path = im_path.replace('videos', 'annotations')
    txt_path = os.path.dirname(txt_path)+'.txt'
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    points = []

    for line in lines:
        line = line.strip()
        line = line.split(' ')
        # print(line)
        annotation = line[1:]
        # print(annotation)
        # print(line[0])
        if int(line[0]) == int(file_path):
            # print(line)
            # print('found', len(annotation))
            for i in range(0, len(annotation), 7):
                y_anno = min(int(float(annotation[i+6])), int(im_h))
                x_anno = min(int(float(annotation[i+5])), int(im_w))
                points.append([y_anno, x_anno])
                # print([y_anno, x_anno])
                # points.append([int(annotation[i+6]), int(annotation[i+5])])
    
    # for i in range(0, len(anno_list)):
    #     y_anno = min(
    #         int(
    #             (
    #                 anno_list[i]['shape_attributes']['y']
    #                 + anno_list[i]['shape_attributes']['height'] / 2
    #             )
    #         ),
    #         int(im_h),
    #     )
    #     x_anno = min(
    #         int(
    #             (
    #                 anno_list[i]['shape_attributes']['x']
    #                 + anno_list[i]['shape_attributes']['width'] / 2
    #             )
    #         ),
    #         int(im_w),
    #     )
    #     points.append([y_anno, x_anno])
    points = np.array(points)
    # print(points.shape)
    # mat_path = os.path.join(
    #     os.path.dirname(im_path).replace('images', 'ground_truth'),
    #     'GT_' + os.path.basename(im_path).replace('jpg', 'mat'),
    # )
    # points = loadmat(mat_path)['locations'].astype(np.float32)
    idx_mask = (
        (points[:, 0] >= 0)
        * (points[:, 0] <= im_w)
        * (points[:, 1] >= 0)
        * (points[:, 1] <= im_h)
    )
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument(
        '--origin-dir',
        default='../../VSCrowd/videos/videos',
        help='original data directory',
    )
    parser.add_argument(
        '--data-dir',
        default='./VSCrowd/preprocessed_data',
        help='processed data directory',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 360
    max_size = 1080
    for phase in ['train_*', 'test_*']:
        sub_dir = os.path.join(args.origin_dir, phase)
        paths = glob(os.path.join(sub_dir, '*jpg'))
        paths.sort()
        print(paths)
        if phase == 'train_*':
            i = 0
            train_paths = paths[:int(len(paths) * 0.9)]
            val_paths = paths[int(len(paths) * 0.9):]
            # for paths in [train_paths, val_paths]:
            #     sub_phase = 'train' if paths == train_paths else 'val'
            sub_save_dir = os.path.join(save_dir, 'train')
            if not os.path.exists(sub_save_dir):
               os.makedirs(sub_save_dir)
            for path in paths:
                sub_save_dir = os.path.join(save_dir, 'train')
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                im_path = path
                name = os.path.dirname(im_path).split('/')[-1]+'_'+os.path.basename(im_path)
                if os.path.exists(os.path.join(sub_save_dir, name)):
                    continue
                im, points = generate_data(im_path)
                    # if points.shape[0] > 3:
                        # continue
                    # print(points.shape)
                # if sub_phase == 'train':
                dis = find_dis(points)
                points = np.concatenate([points, dis], axis=1)
                im.save(os.path.join(sub_save_dir, name))
                np.save(
                    os.path.join(sub_save_dir, name.replace('.jpg', '.npy')), points
                )

                print('{}: {}/{}'.format('train', i, len(paths)))
                i += 1       
        # elif phase == 'val_*':  
        #     i=0
        #     sub_save_dir = os.path.join(save_dir, 'val')
        #     if not os.path.exists(sub_save_dir):
        #         os.makedirs(sub_save_dir)
        #     for path in paths:
        #         sub_save_dir = os.path.join(save_dir, 'val')
        #         if not os.path.exists(sub_save_dir):
        #             os.makedirs(sub_save_dir)
        #         im_path = path
        #         name = os.path.dirname(im_path).split('/')[-1]+'_'+os.path.basename(im_path)
        #         im, points = generate_data(im_path)
        #             # if points.shape[0] > 3:
        #                 # continue
        #             # print(points.shape)
        #             # if sub_phase == 'train':
        #         # dis = find_dis(points)
        #         # points = np.concatenate([points, dis], axis=1)
        #         im.save(os.path.join(sub_save_dir, name))
        #         np.save(
        #             os.path.join(sub_save_dir, name.replace('.jpg', '.npy')), points
        #         )

        #         print('{}: {}/{}'.format('val', i, len(paths)))
        #         i += 1     
        else:
            i = 0
            for path in paths:
                sub_save_dir = os.path.join(save_dir, 'test')
                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)
                im_path = path
                name = os.path.dirname(im_path).split('/')[-1]+'_'+os.path.basename(im_path)
                # im_path = os.path.join(args.origin_dir, im_path)
                im, points = generate_data(im_path)
                im.save(os.path.join(sub_save_dir, name))
                np.save(
                    os.path.join(sub_save_dir, name.replace('.jpg', '.npy')), points
                )
                print('\r{}: {}/{}'.format(phase, i, len(paths)))
                i += 1

    # for phase in ['train', 'test']:
    #     sub_dir = os.path.join(args.origin_dir, phase)
    #     if phase == 'train':
    #         sub_phase_list = ['train', 'val']
    #         for sub_phase in sub_phase_list:
    #             sub_save_dir = os.path.join(save_dir, sub_phase)
    #             if not os.path.exists(sub_save_dir):
    #                 os.makedirs(sub_save_dir)
    #             with open(os.path.join(args.origin_dir, sub_phase + '.json'), 'r') as f:
    #                 paths = json.load(f)
    #             i = 0
    #             for path in paths:
    #                 im_path = path
    #                 name = os.path.basename(im_path)
    #                 im_path = os.path.join(args.origin_dir, im_path)
    #                 im, points = generate_data(im_path)
    #                 if sub_phase == 'train':
    #                     dis = find_dis(points)
    #                     points = np.concatenate([points, dis], axis=1)
    #                 im.save(os.path.join(sub_save_dir, name))
    #                 np.save(
    #                     os.path.join(sub_save_dir, name.replace('.jpg', '.npy')), points
    #                 )
    #                 print('\r{}: {}/{}'.format(sub_phase, i, len(paths)), end='')
    #                 i += 1
    #             print()
    #     else:
    #         sub_save_dir = os.path.join(save_dir, 'test')
    #         if not os.path.exists(sub_save_dir):
    #             os.makedirs(sub_save_dir)
    #         with open(os.path.join(args.origin_dir, 'test.json'), 'r') as f:
    #             paths = json.load(f)
    #         im_list = paths
    #         # im_list = glob(os.path.join(sub_dir, '*jpg'))
    #         i = 0
    #         for im_path in im_list:
    #             # im_path = os.path.join('../../nas-public-linkdata/ds/dronebird', im_path)
    #             name = os.path.basename(im_path)
    #             # print(name)
    #             path = os.path.join(args.origin_dir, im_path)
    #             im, points = generate_data(path)
    #             im_save_path = os.path.join(sub_save_dir, name)
    #             im.save(im_save_path)
    #             gd_save_path = im_save_path.replace('jpg', 'npy')
    #             np.save(gd_save_path, points)
    #             print('\r{}: {}/{}'.format(phase, i, len(im_list)), end='')
    #             i += 1
    #         print()
