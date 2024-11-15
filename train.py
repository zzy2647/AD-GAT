
from logging import CRITICAL
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from thop import profile

import torch.utils.data as data
from Datasets.DatasetADNI import get_dataset, collate_fn_name, ToTensor
from Datasets.DatasetADNI import DatasetADNI
from Models.VariableGAT import VariableGAT
from Models.ConvMethods import VGG16, ResNet, Resnext, VIT_b_16, H_ConvNet, AlzNet #, AlexNet, MobileNet
from Models.VAN import VAN_b1,VAN_b0
# from Models.convGNN import ConvGNN
from Models.convGNN_test import ConvGNN
from Models.twin_gvt import pcpvt_base_v0, alt_gvt_small
from Models.ConvNeXT import convnext_tiny,convnext_small
from Models.inceptionv3 import Inception
from Models.tresnet import TResnetM
from Models.DSCNN import DSCNN
import random

 
def set_seed(seed=666):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def step_lr(epoch, lr):
    learning_rate = lr
    if epoch < 10:
        learning_rate = lr
    elif epoch % 20 == 0:
        learning_rate = lr * 0.5
    return learning_rate
def train():
    train_dataloader, val_dataloader, test_dataloader, num_classes = get_dataset(Batch)
    model = ModelName(num_classes = num_classes).to(device)
    # model = ModelName().to(device)
    input = torch.randn(1, 1, 224, 224).to(device)
    flops, params = profile(model, inputs=(input,))
    model_info = "params:%.2f | flops:%.2f" % (params / (1000 ** 2), flops / (1000 ** 3))
    print(model_info)
    with open(ProcessingPath, 'a') as f:
            f.write(model_info+'\n')
    
    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.CrossEntropyLoss()
    # VariableGAT
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # best_model lr=0.01
    # VGG16
    # optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    # train
    best_acc = 0
    # optimizer.zero_grad()
    lr = 0.001
    for epoch in range(1, Epoch+1):

        train_loss,train_loss1,train_loss2 = 0, 0, 0
        train_num = 0

        model.train()
        lr = step_lr(epoch,lr)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for i, data in enumerate(tqdm(train_dataloader, desc='training')):
            batch_imgs, batch_targets = data
            batch_imgs = batch_imgs.to(device)
            batch_targets = batch_targets.to(device)

            batch_predict = model(batch_imgs)
            loss, ANloss, AMloss  = split_out_label(batch_predict, batch_targets, w1 = 0.0, w2 = 1.0, criterion = criterion, criterion1 = criterion1) #1 0.4
            # GoogLeNet 需要加入下面这一行
            # batch_predict = batch_predict.logits
            # loss = criterion(batch_predict, batch_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            train_loss += loss.item() * batch_targets.shape[0]
            train_loss1 += ANloss.item() * batch_targets.shape[0]
            train_loss2 += AMloss.item() * batch_targets.shape[0]
            train_num += batch_targets.shape[0]

        train_loss, train_loss1, train_loss2 = train_loss/train_num, train_loss1/train_num, train_loss2/train_num

        acc, conf_matrix, val_loss = test(model, val_dataloader)
        info = f"epoch:{epoch}/{Epoch}, " \
               f"train_loss:{train_loss:.7f}, " \
               f"train_loss1:{train_loss1:.7f}, " \
               f"train_loss2:{train_loss2:.7f}, " \
               f"test_loss:{val_loss:.5f}, " \
               f"val_acc:{acc:.5f}"

        print(info)
        with open(ProcessingPath, 'a') as f:
            f.write(info+'\n')
        with open(LossPath, 'a') as f:
            f.write(f"{train_loss}\t{val_loss}\n")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(BasePath, f'parameter_best.pth'))
            save_confusion(conf_matrix, acc)
            with open(ProcessingPath, 'a') as f:
                f.write(f"save best model parameter, epoch {epoch} \n")
    torch.save(model.state_dict(), os.path.join(BasePath, f'parameter_last.pth'))


def test(model, loader):
    model.eval()
    predict = []
    targets = []
    test_loss = 0
    test_loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    for i, data in enumerate(tqdm(loader, desc='val')):
        batch_imgs, batch_targets = data
        batch_imgs = batch_imgs.to(device)
        batch_targets = batch_targets.to(device)
        batch_predict = model(batch_imgs)
        test_loss += test_loss_func(batch_predict, batch_targets).cpu().item()
        predict.append(torch.max(batch_predict, 1)[1].cpu().numpy())
        targets.append(batch_targets.cpu().numpy())
    predict = np.concatenate(predict)
    targets = np.concatenate(targets)

    acc, conf_matrix = calculate_indicator(predict, targets)
    test_loss = test_loss / targets.size

    return acc, conf_matrix, test_loss

def predict(num_workers = 8):
    root = "/home/hp-video/Documents/zhangzhengyang/AD-GAT/ADNI_Dataset"
    val_dataset = DatasetADNI(os.path.join(root, "Test"), ToTensor(), ret_name=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=Batch, shuffle=False, num_workers=num_workers,
                                     collate_fn=collate_fn_name)
    model = ModelName(num_classes = len(val_dataset.CLASS_NAME)).to(device)
    model.load_state_dict(torch.load(load_path),strict = False)

    model.eval()
    predict = []
    targets = []
    name_list = []
    for i, batch_data in enumerate(tqdm(val_dataloader, desc='val')):
        names, batch_imgs, batch_targets = batch_data
        batch_imgs = batch_imgs.to(device)
        batch_targets = batch_targets.to(device)
        batch_predict = model(batch_imgs)

        predict.append(torch.max(batch_predict, 1)[1].cpu().numpy())
        targets.append(batch_targets.cpu().numpy())
        name_list.append(names)
    # print(name_list)
    predict = np.concatenate(predict)
    targets = np.concatenate(targets)
    # print(predict)
    # for i, pre in enumerate(predict):
    #     if pre != targets[i]:
    #         print(name_list[i])

    acc, conf_matrix = calculate_indicator(predict, targets)
    print(acc)
    print(conf_matrix)



def calculate_indicator(pre, true):
    assert len(pre) == len(true)
    conf_matrix = confusion_matrix(y_true=true, y_pred=pre)

    diag = conf_matrix.diagonal()
    sample_num = len(pre)
    acc = diag.sum() / sample_num

    return acc, conf_matrix


def save_confusion(conf_matrix, acc):
    with open(os.path.join(BasePath, 'conf_matrix.txt'), 'w') as f:
        for index in range(len(conf_matrix)):
            f.write(f'{conf_matrix[index][0]}\t\t{conf_matrix[index][1]}\t\t{conf_matrix[index][2]}\n')
        f.write(f'{acc}\n')

def split_out_label(y, yt, w1, w2, criterion, criterion1):
    # criterion1 = nn.BCEWithLogitsLoss()
    # m = nn.Sigmoid()
    y_sum = y[:,0]+y[:,1]
    
    y1 = torch.stack([y_sum,y[:,2]], dim = 1)
    # y2 = torch.cat([y[:,:2],torch.FloatTensor(len(yt),1).fill_(0.).to(device)],dim=1)
    # print(y1,y2)
    # y1 = m(y1.to(device))[:,1]
    # y2 = y2.to(device)
    yt_1 = yt.clone()
    
    for i,label in enumerate(yt_1):
        if label != torch.tensor(2):
            yt_1[i] = 0
        else:
            yt_1[i] = 1

    loss1 = torch.tensor(w1).to(device) * criterion(y1, yt_1) #.float()
    loss2 = torch.tensor(w2).to(device) * criterion1(y, yt)
    # loss3 = torch.tensor(w2).to(device) * criterion(y2, yt)
    loss = loss1 + loss2
    
    return loss ,loss1, loss2

if __name__ == '__main__':
    
    # input = torch.randn(1, 1, 224, 224)
    # model1 = ResNet(num_classes=3)
    # flops, params = profile(model1, inputs=(input,))
    # model_info = "params:%.2f | flops:%.2f" % (params / (1000 ** 2), flops / (1000 ** 3))
    # print(model_info)
    
    Epoch = 100
    device = "cuda:0"
    Batch = 4
    ModelName = ConvGNN #VariableGAT GoogLeNet VGG16 van_b1(in_chans=1, num_classes=3) H_ConvNet ConvGNN convnext_tiny

    BasePath = './checkpoint/ConvGNN_test'
    LossPath = BasePath + '/loss.txt'
    ProcessingPath = BasePath + '/processing.txt'
    ResultPath = BasePath + '/result.txt'
    load_path = './checkpoint/ConvGNN_test/parameter_best.pth'
    # os.makedirs(BasePath, exist_ok=True)
    set_seed(666)
    train()
    predict()