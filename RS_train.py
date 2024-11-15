import torch
import numpy as np
from train import train
from train_kflod import train_kflod,set_seed,predict
from train_kflod import get_imgs_label
from Models.ConvMethods import VGG16, ResNet, Resnext, VIT_b_16, H_ConvNet, AlzNet #, AlexNet, MobileNet
from Models.VAN import VAN_b1,VAN_b0
from Models.convGNN import ConvGNN
# from Models.convGNN_test import ConvGNN
from Models.twin_gvt import pcpvt_base_v0, alt_gvt_small
from Models.ConvNeXT import convnext_tiny,convnext_small
from Models.inceptionv3 import Inception
from Models.tresnet import TResnetM
from Models.DSCNN import DSCNN
class Configs(object):
    def __init__(self):
        self.ModelName = TResnetM
        self.model = 'TResnetM'
        self.Epoch = 120
        self.test_batchsize = 8
        self.device = "cuda:0"
        self.BasePath = f'./checkpoint/{self.model}'
        self.LossPath = self.BasePath + '/loss.txt'
        self.ProcessingPath = self.BasePath + '/processing.txt'
        self.ResultPath = self.BasePath + '/result.txt'
        self.load_path = f'./checkpoint/{self.model}/parameter_best.pth' #f'./checkpoint/{self.model}/parameter_best.pth'
        self.best_path = './checkpoint/ConvGNN/last/kfold_parameter_best.pth' #'./checkpoint/ConvGNN/kfold_parameter_best.pth'
    def get_members(self):
        return vars(self)


# 定义超参数空间
hyperparams = {
    'lr': [0.001],
    'batch_size': [4],
    # 'hidden_size': [32, 64, 128, 256],
    # 'num_layers': [1, 2, 3, 4],
    # 'dropout': np.linspace(0.1, 0.5, 5)
}

best_accuracy = 0
best_hyperparams = {}
imgs_train, labels_train = get_imgs_label()
imgs_test, labels_test = get_imgs_label(dir="/home/hp-video/Documents/zhangzhengyang/data/ADNI_Dataset3/Test2/")
# Epoch = 100
num_iterations = 1
# device = "cuda:0"
# ModelName = ConvGNN #VariableGAT GoogLeNet VGG16 van_b1(in_chans=1, num_classes=3) H_ConvNet ConvGNN convnext_tiny

# BasePath = f'./checkpoint/{str(ModelName)}'
# LossPath = BasePath + '/loss.txt'
# ProcessingPath = BasePath + '/processing.txt'
# ResultPath = BasePath + '/result.txt'
# load_path = f'./checkpoint/{str(ModelName)}/parameter_best.pth'
# os.makedirs(BasePath, exist_ok=True)
set_seed(666)
'''
for i in range(num_iterations):
    # 随机采样超参数组合
    hyperparams_sample = {}
    for key, value in hyperparams.items():
        hyperparams_sample[key] = np.random.choice(value)

    # 模型训练和评估
    accuracy = train_kflod(imgs_train,labels_train,hyperparams_sample,config=Configs(),best_acc_kflod=best_accuracy)

    # 记录最优超参数组合
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_hyperparams = hyperparams_sample

print(best_hyperparams)
'''
predict(imgs_test,labels_test,num_workers=8,config=Configs())