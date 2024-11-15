from logging import CRITICAL
import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from Datasets.DatasetADNI import get_dataset
from Models.VariableGAT import VariableGAT
from Models.ConvMethods import VGG16, GoogLeNet, ResNet, VIT_b_16, H_ConvNet #, AlexNet, MobileNet
from Models.VAN import VAN_b1
# from Models.convGNN import ConvGNN
from Models.convGNN_test import ConvGNN_test
from Models.convGNN import ConvGNN
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

def train():
    train_dataloader, val_dataloader, test_dataloader, num_classes = get_dataset(Batch)
    model = ModelName(num_classes = num_classes).to(device)
    # weights = torch.tensor([321,1100,611], dtype=torch.float32)
    # weights = [max(weights)/x for x in weights]
    # criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights).to(device))
    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.CrossEntropyLoss()
    # VariableGAT
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
    # VGG16
    # optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    # train
    best_acc = 0
    optimizer.zero_grad()
    for epoch in range(1, Epoch+1):

        train_loss,train_loss1,train_loss2 = 0, 0, 0
        train_num = 0

        model.train()
        for i, data in enumerate(tqdm(train_dataloader, desc='training')):
            batch_imgs, batch_targets = data
            batch_imgs = batch_imgs.to(device)
            batch_targets = batch_targets.to(device)

            batch_predict = model(batch_imgs)
            loss, ANloss, AMloss  = split_out_label(batch_predict, batch_targets, w1 = 1., w2 = 0.4, criterion = criterion, criterion1 = criterion1)
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
    y2 = torch.cat([y[:,:2],torch.FloatTensor(len(yt),1).fill_(0.).to(device)],dim=1)
    # print(y1,y2)
    # y1 = m(y1.to(device))[:,1]
    y2 = y2.to(device)
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
    Epoch = 200
    device = "cuda:0"
    Batch = 8
    ModelName = ConvGNN_test #VariableGAT GoogLeNet VGG16 van_b1(in_chans=1, num_classes=3) H_ConvNet ConvGNN

    LR = 0.00005
    BasePath = './checkpoint/ConvGNN_best'
    LossPath = BasePath + '/loss.txt'
    ProcessingPath = BasePath + '/processing.txt'
    ResultPath = BasePath + '/result.txt'

    os.makedirs(BasePath, exist_ok=True)
    set_seed(666)
    train()
