import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg_c import vgg19_trans
import argparse
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error
import scipy.io as sio
args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='./preprocessed_data',
                        help='training data directory')
    parser.add_argument('--save-dir', default='model/man_best_model.pth',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args
def get_seq_class(seq, set):
    backlight = ['DJI_0021', 'DJI_0022', 'DJI_0032', 'DJI_0202', 'DJI_0339', 'DJI_0340']
    # cloudy = ['DJI_0519', 'DJI_0554']
    
    # uhd = ['DJI_0332', 'DJI_0334', 'DJI_0339', 'DJI_0340', 'DJI_0342', 'DJI_0343', 'DJI_345', 'DJI_0348', 'DJI_0519', 'DJI_0544']

    fly = ['DJI_0177', 'DJI_0174', 'DJI_0022', 'DJI_0180', 'DJI_0181', 'DJI_0200', 'DJI_0544', 'DJI_0012', 'DJI_0178', 'DJI_0343', 'DJI_0185', 'DJI_0195']

    angle_90 = ['DJI_0179', 'DJI_0186', 'DJI_0189', 'DJI_0191', 'DJI_0196', 'DJI_0190']

    mid_size = ['DJI_0012', 'DJI_0013', 'DJI_0014', 'DJI_0021', 'DJI_0022', 'DJI_0026', 'DJI_0028', 'DJI_0028', 'DJI_0030', 'DJI_0028', 'DJI_0030', 'DJI_0034','DJI_0200', 'DJI_0544']

    light = 'sunny'
    bird = 'stand'
    angle = '60'
    size = 'small'
    # resolution = '4k'
    if seq in backlight:
        light = 'backlight'
    if seq in fly:
        bird = 'fly'
    if seq in angle_90:
        angle = '90'
    if seq in mid_size:
        size = 'mid'

    # if seq in uhd:
    #     resolution = 'uhd'
    
    count = 'sparse'
    loca = sio.loadmat(os.path.join('../../ds/dronebird/', set, 'ground_truth', 'GT_img'+str(seq[-3:])+'000.mat'))['locations']
    if loca.shape[0] > 150:
        count = 'crowded'
    return light, angle, bird, size


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 1024, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=0, pin_memory=False)

    preds = [[] for i in range(10)]
    gts = [[] for i in range(10)]
    device = torch.device('cuda')
    model = vgg19_trans()
    model.to(device)
    model.eval()
    if args.save_dir.endswith('.pth'):
        model.load_state_dict(torch.load(args.save_dir, device))
    else:
        model.load_state_dict(torch.load(args.save_dir, device)['model_state_dict'])
    epoch_minus = []
    it = 0
    for inputs, count, name in dataloader:
        # print(name)
        seq = int(name[0][3:6])
        seq = 'DJI_' + str(seq).zfill(4)
        light, angle, bird, size = get_seq_class(seq, 'test')
        inputs = inputs.to(device)
        b, c, h, w = inputs.shape
        h, w = int(h), int(w)
        assert b == 1, 'the batch size should equal to 1 in validation mode'
        input_list = []
        if h >= 3584 or w >= 3584:
            h_stride = int(math.ceil(1.0 * h / 3584))
            w_stride = int(math.ceil(1.0 * w / 3584))
            h_step = h // h_stride
            w_step = w // w_stride
            for i in range(h_stride):
                for j in range(w_stride):
                    h_start = i * h_step
                    if i != h_stride - 1:
                        h_end = (i + 1) * h_step
                    else:
                        h_end = h
                    w_start = j * w_step
                    if j != w_stride - 1:
                        w_end = (j + 1) * w_step
                    else:
                        w_end = w
                    input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
            with torch.set_grad_enabled(False):
                pre_count = 0.0
                for idx, input in enumerate(input_list):
                    output = model(input)[0]
                    pre_count += torch.sum(output)
            
            res = count[0].item() - pre_count.item()
            epoch_minus.append(res)
        else:
            with torch.set_grad_enabled(False):
                outputs = model(inputs)[0]
                res = count[0].item() - torch.sum(outputs).item()
                epoch_minus.append(res)
        gt_e = count[0].item()
        pred_e = pre_count.item()
        count = 'crowded' if gt_e > 150 else 'sparse'

        if light == 'sunny':
            preds[0].append(pred_e)
            gts[0].append(gt_e)
        elif light == 'backlight':
            preds[1].append(pred_e)
            gts[1].append(gt_e)
        # else:
        #     preds[2].append(pred_e)
        #     gts[2].append(gt_e)
        if count == 'crowded':
            preds[2].append(pred_e)
            gts[2].append(gt_e)
        else:
            preds[3].append(pred_e)
            gts[3].append(gt_e)
        if angle == '60':
            preds[4].append(pred_e)
            gts[4].append(gt_e)
        else:
            preds[5].append(pred_e)
            gts[5].append(gt_e)
        if bird == 'stand':
            preds[6].append(pred_e)
            gts[6].append(gt_e)
        else:
            preds[7].append(pred_e)
            gts[7].append(gt_e)
        if size == 'small':
            preds[8].append(pred_e)
            gts[8].append(gt_e)
        else:
            preds[9].append(pred_e)
            gts[9].append(gt_e)
        it += 1
        print('\r{:>{}}/{}: {}, pred: {}, gt: {}'.format(it, len(str(len(dataloader))), len(dataloader), res, pred_e, gt_e), end='')
    print()
    # 输出至txt文件
    with open('result.txt', 'w') as f:
        f.write('max: {}, min: {}\n'.format(max(np.abs(epoch_minus)), min(np.abs(epoch_minus))))
        print('max: {}, min: {}'.format(max(np.abs(epoch_minus)), min(np.abs(epoch_minus))))
        epoch_minus = np.array(epoch_minus)
        mse = np.sqrt(np.mean(np.square(epoch_minus)))
        mae = np.mean(np.abs(epoch_minus))
        log_str = 'mae {}, mse {}\n'.format(mae, mse)
        print(log_str)
        f.write(log_str)
        attri = ['sunny', 'backlight', 'crowded', 'sparse', '60', '90', 'stand', 'fly', 'small', 'mid']
        for i in range(10):
            if len(preds[i]) == 0:
                continue
            print('{}: MAE:{}. RMSE:{}.'.format(attri[i], mean_absolute_error(preds[i], gts[i]), np.sqrt(mean_squared_error(preds[i], gts[i]))))
            f.write('{}: MAE:{}. RMSE:{}.\n'.format(attri[i], mean_absolute_error(preds[i], gts[i]), np.sqrt(mean_squared_error(preds[i], gts[i]))))
