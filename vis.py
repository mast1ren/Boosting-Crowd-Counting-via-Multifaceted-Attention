import torch
from models import vgg_c
import argparse
from datasets.crowd import Crowd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import scipy
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import glob

args = None
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='./preprocessed_data',
                        help='training data directory')
    parser.add_argument('--save-dir', default='model/man_best_model.pth',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args

def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        for index,layer in enumerate(feature_extractor):
            x = layer(x)
            if k == index:
                return x


def main():
    args = parse_args()
    model = vgg_c.vgg19_trans()
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    if args.save_dir.endswith('.pth'):
        model.load_state_dict(torch.load(args.save_dir, device))
    else:
        model.load_state_dict(torch.load(args.save_dir, device)['model_state_dict'])

    print(dict(model.features.named_children()))
    
    features = model.features

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 1024, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=0, pin_memory=False)
    paths = glob.glob(os.path.join('../../ds/dronebird/vis', '*.jpg'))
    last_seq = -1
    target_layers = [features[-1]]
    with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
        targets = [ClassifierOutputTarget(3)]

        for path in paths:
            inputs = Image.open(path).convert('RGB')
            inputs = np.array(inputs)

            seq = os.path.basename(path).split('_')[1]

            if seq != last_seq:
                last_seq = seq
                print('Processing %s' % seq)
            else:
                continue
            inputs = torch.from_numpy(inputs).permute(2, 0, 1).unsqueeze(0).float()
            inputs = inputs.to(device)
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            

            # input_tensor = inputs


            # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # grayscale_cam = grayscale_cam[0, :]
            # visualization = show_cam_on_image(inputs, grayscale_cam, use_rgb=True)
            # plt.imshow(visualization)
            feature_map = get_k_layer_feature_map(features, 34, inputs)
            feature_map = feature_map.squeeze(0)
            feature_map = feature_map.cpu().numpy()
            feature_map_num = feature_map.shape[0]
            row_num = int(np.ceil(np.sqrt(feature_map_num)))
            plt.figure()
            for index in range(1, feature_map_num+1):
                plt.subplot(row_num, row_num, index)
                plt.imshow(feature_map[index-1], cmap='gray')
                plt.axis('off')
    
            # plt.imsave("./features/"+str(index)+".png", feature_map[index-1])
            plt.savefig("./features/feature_"+str(seq)+".svg", format='svg', dpi=1200)
        # # plt.show()

        # gt_e = count[0].item()
        # pred_e = pre_count.item()
        # break

if __name__ == '__main__':
    main()
