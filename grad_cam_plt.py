import argparse
from ast import arg
import os
from matplotlib import pyplot as plt
import cv2

from torch import device
import torch
from Datasets.DatasetADNI import get_dataset
from Models.convGNN import ConvGNN
from Models.ConvMethods import VGG16, ResNet, Resnext, VIT_b_16, H_ConvNet #, AlexNet, MobileNet
from Models.VAN import VAN_b1,VAN_b0
from Models.twin_gvt import pcpvt_small_v0, alt_gvt_small
from Models.ConvNeXT import convnext_tiny,convnext_small
from Models.inceptionv3 import Inception
from Models.tresnet import TResnetM
from Models.DSCNN import DSCNN

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, EigenGradCAM, LayerCAM 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from einops import rearrange
from PIL import Image
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--device',type=str , default = "cuda:0")
    parser.add_argument(
        '--image-path',
        type=str,
        default='./ADNI_Dataset2/cam/',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    parser.add_argument(
        '--cnn-or-vit',
        action='store_true',
        default=False,
        help='CNN Model or Vit Model')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

def reshape_transform(tensor_x):

    result = tensor_x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
    return result

def reshape_PGNN(tensor_x):
    result = rearrange(tensor_x[:, 1:, :], 'b (h w) c -> b h w c',h=7)

    # Bring the channels to the first dimension,
    # like in CNNs.
    # [batch_size, H, W, C] -> [batch, C, H, W]
    result = result.permute(0, 3, 1, 2)
    return result

if __name__ == '__main__':
    
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    #参数设置
    model_str = 'ConvGNN_test2'  #ConvGNN ResNet Convnext VAN
    ModelName = ConvGNN
    load_path = './checkpoint/ConvGNN/0.9447852760736196_kfold_parameter_best.pth'

    #加载模型
    model = ModelName(num_classes=3).to(args.device)
    model.load_state_dict(torch.load(load_path),strict=False)
    print(model)
    model.eval()
    # print(model.convNet.stages[3][1].dwconv)

    #设置选择的层
    target_layers = [model.convNet.stages[3][0]]
    # target_layers = [model.dwconv2]
    # target_layers = [model.van_b0.block4[1]]
    # target_layers = [model.resnet.layer4]
    # target_layers =[model.graphNet.layer[0][1]]

    #生成热力图图片

    train_dataloader, val_dataloader, test_dataloader, num_classes = get_dataset(1,root='/home/hp-video/Documents/zhangzhengyang/AD-GAT/ADNI_Dataset2')
    for step, data in enumerate(test_dataloader):
        img_name, batch_imgs, batch_targets = data
        # print(img_name)
        img_path = os.path.join(args.image_path,img_name[0])
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]   # 1是读取rgb
                                                    #imread返回从指定路径加载的图像
        rgb_img = cv2.imread(img_path, 1) #imread()读取的是BGR格式
        rgb_img = np.float32(rgb_img) / 255
        img_pad = np.pad(rgb_img, ((21, 21), (3, 3),(0,0)), 'constant', constant_values=0) 
        
        if args.cnn_or_vit:
            cam = methods[args.method](model=model, target_layers=target_layers, use_cuda=args.use_cuda)
            print('mode1')
        elif args.method == "ablationcam":
            cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_PGNN,
                                   ablation_layer=AblationLayerVit())
            print("mode2")
        else:
            cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                #    reshape_transform=reshape_PGNN
                                   )
            print("mode3")

        

        targets = None

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=batch_imgs, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img_pad, grayscale_cam, use_rgb=True)
        # cv2.imwrite(f'cam_img/first_try.jpg', visualization)
        if not os.path.exists(f'./cam_img/{model_str}/'):
            os.makedirs(f'./cam_img/{model_str}/')
        plt.imsave(f'./cam_img/{model_str}/{img_name[0].split(".")[0]}.jpg', visualization)  
    
    def padimage():
        dir = './cam_img/compose_fig/origin_fig/'
        for images in os.listdir(dir):
            img_path = os.path.join(dir,images)
            rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]   # 1是读取rgb
                                                    #imread返回从指定路径加载的图像
            rgb_img = cv2.imread(img_path, 1) #imread()读取的是BGR格式
            rgb_img = np.float32(rgb_img) / 255
            img_pad = np.pad(rgb_img, ((21, 21), (3, 3),(0,0)), 'constant', constant_values=0) 
            plt.imsave(f'./cam_img/compose_fig/origin_fig/{images.split(".")[0]}.jpg', img_pad)


    # padimage()




