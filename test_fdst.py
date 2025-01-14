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
    parser.add_argument('--data-dir', default='./VSCrowd/preprocessed_data',
                        help='training data directory')
    # parser.add_argument('--save-dir', default='model/0128-145952/best_model.pth',
    parser.add_argument('--save-dir', default='model/0226-104805/126_ckpt.tar',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 1024, 16, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=0, pin_memory=False)

    preds = []
    gts = []
    names = []
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
                pre_count = torch.sum(outputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_minus.append(res)
        gt_e = count[0].item()
        pred_e = pre_count.item()
        preds.append(pred_e)
        gts.append(gt_e)
        names.append(name[0])
        
        it += 1
        print('\r{:>{}}/{}: {}, pred: {}, gt: {}'.format(it, len(str(len(dataloader))), len(dataloader), res, pred_e, gt_e), end='')
    print()
    # 输出至txt文件
    
    with open('result.txt', 'w') as f:
        for (name, pred, gt) in zip(names, preds, gts):
            f.write('{}: pred: {}, gt: {}\n'.format(name, pred, gt))

        f.write('max: {}, min: {}\n'.format(max(np.abs(epoch_minus)), min(np.abs(epoch_minus))))
        print('max: {}, min: {}'.format(max(np.abs(epoch_minus)), min(np.abs(epoch_minus))))
        epoch_minus = np.array(epoch_minus)
        mse = np.sqrt(np.mean(np.square(epoch_minus)))
        mae = np.mean(np.abs(epoch_minus))
        log_str = 'mae {}, mse {}\n'.format(mae, mse)
        print(log_str)
        f.write(log_str)
