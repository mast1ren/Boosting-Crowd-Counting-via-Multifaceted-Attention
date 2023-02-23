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
import cv2
from torchvision import transforms

args = None
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

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
    device = torch.device('cuda')
    model.to(device)
    if args.save_dir.endswith('.pth'):
        model.load_state_dict(torch.load(args.save_dir, device))
    else:
        model.load_state_dict(torch.load(args.save_dir, device)['model_state_dict'])

    model.eval()
    # print(dict(model.features.named_children()))
    
    features = model.features

    datasets = Crowd(os.path.join(args.data_dir, 'train'), 512, 16, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=0, pin_memory=False)
    paths = glob.glob(os.path.join('../../ds/dronebird/vis', '*.jpg'))
    last_seq = -1
    target_layers = [features[34]]
    #  
    seqs = ['021', '012', '174','001', '002']
    sets = ['val']
    with torch.no_grad():
        for path in paths:
            seq = path.split('/')[-1].split('_')[1].split('.')[0]
            set = path.split('/')[-1].split('_')[0]
            if seq != last_seq and (set in sets):
                last_seq = seq
                print("Processing %s" % seq)
            else:
                continue
            
            inputs = transform(Image.open(path).convert("RGB")).unsqueeze(0)
            # inputs = inputs.resize((1920, 1080))
            # inputs = np.array(inputs)
            # inputs = torch.from_numpy(inputs)
            # inputs = torch.from_numpy(inputs).permute(2, 0, 1).unsqueeze(0).float()
            inputs = inputs.to(device)
            outputs = model(inputs)[0].cpu()
            # print(torch.sum(outputs).item())
            den = np.array(outputs[0,0])
            # print(den.sum())
            den = cv2.resize(den, (inputs.shape[3], inputs.shape[2]), interpolation=cv2.INTER_CUBIC)/((inputs.shape[3]/den.shape[1])*(inputs.shape[2]/den.shape[0]))
            print(den.sum())


            # feature_map = get_k_layer_feature_map(features, 34, inputs)
            # feature_map = feature_map.squeeze(0)
            # feature_map = feature_map.cpu().numpy()
            # feature_map_num = feature_map.shape[0]

            # row_num = int(np.ceil(np.sqrt(feature_map_num)))
            # plt.figure()
            # for index in range(1, feature_map_num+1):
            #     plt.subplot(row_num, row_num, index)
            #     plt.imshow(feature_map[index-1], cmap='gray')
            #     plt.axis('off')
            #     if index == 1:
            #         plt.imsave("./vis/"+str(seq)+'_'+str(index)+".jpg", feature_map[index-1])
            # plt.savefig("./vis/"+str(seq)+".jpg", dpi=1200)
            pred = den.sum()
            den = np.expand_dims(den, 0)
            den = (den-den.min())/(den.max()-den.min()+1e-10)

            input = np.array(inputs[0].cpu())
            input = (input - input.min()) / (input.max() - input.min()+1e-10)
            # print(den.shape, input.shape)
            # print(den.max(), den.min(), input.max(), input.min())
            vis = np.transpose((input/2+den/2),(1,2,0))
            # print(vis)

            model_name = os.path.basename(args.save_dir)[:3]
            plt.imsave("./vis/"+str(model_name)+ '_'+str(seq) + "_" + str(int(pred))+"_den.jpg", vis)

    # with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
    #         # targets = [ClassifierOutputTarget(3)]
    #         targets = None

    #         # for path in paths:
    #         for inputs, count, name in dataloader:
    #             # inputs = Image.open(path).convert('RGB')
    #             # inputs = inputs.resize((1920,1080))
    #             # inputs = np.array(inputs)
    #             # seq = os.path.basename(path)[3:6]
    #             seq = name[0][3:6]

    #             if seq != last_seq:
    #                 last_seq = seq
    #                 print('Processing %s' % seq)
    #             else:
    #                 continue
    #             # inputs = torch.from_numpy(inputs).permute(2, 0, 1).unsqueeze(0).float()
    #             # inputs = torch.from_numpy(inputs)
    #             inputs = inputs.to(device)
    #             # b, c, h, w = inputs.shape
    #             # h, w = int(h), int(w)
    #             # assert b == 1, 'the batch size should equal to 1 in validation mode'


    #             input_tensor = inputs


    #             grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    #             grayscale_cam = grayscale_cam[0, :]
    #             visualization = show_cam_on_image(inputs, grayscale_cam, use_rgb=True)
    #             plt.imsave('./vis/'+seq+'.jpg', visualization)
                # plt.imshow(visualization)
                # feature_map = get_k_layer_feature_map(features, 34, inputs)
                # feature_map = feature_map.squeeze(0)
                # feature_map = feature_map.cpu().numpy()
                # feature_map_num = feature_map.shape[0]
                # row_num = int(np.ceil(np.sqrt(feature_map_num)))
                # plt.figure()
                # for index in range(1, feature_map_num+1):
                #     plt.subplot(row_num, row_num, index)
                #     plt.imshow(feature_map[index-1], cmap='gray')
                #     plt.axis('off')

                # # plt.imsave("./features/"+str(index)+".png", feature_map[index-1])
                # plt.savefig("./features/feature_"+str(seq)+".svg", format='svg', dpi=1200)
            # # plt.show()

        # gt_e = count[0].item()
        # pred_e = pre_count.item()
        # break

if __name__ == '__main__':
    main()
