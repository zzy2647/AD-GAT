import os
import random
import shutil
import numpy as np


def _split_train_test(train_rate, val_rate, old_path, new_path):
    dir_list = os.listdir(old_path)
    for dir_name in dir_list:
        dir_path = os.path.join(old_path, dir_name)
        image_list = os.listdir(dir_path)
        random.shuffle(image_list)
        image_num = len(image_list)
        train_num = round(image_num * train_rate)
        val_num = round(image_num * val_rate)

        train_list = image_list[0:train_num]
        val_list = image_list[train_num:train_num+val_num]
        test_list = image_list[train_num+val_num:]

        os.makedirs(new_path + "/Train", exist_ok=True)
        os.makedirs(new_path + "/Val", exist_ok=True)
        os.makedirs(new_path + "/Test", exist_ok=True)
        for name in train_list:
            shutil.copy(os.path.join(dir_path, name), f"{new_path}/Train/{dir_name}_{name}")
        for name in val_list:
            shutil.copy(os.path.join(dir_path, name), f"{new_path}/Val/{dir_name}_{name}")
        for name in test_list:
            shutil.copy(os.path.join(dir_path, name), f"{new_path}/Test/{dir_name}_{name}")


def get_train_test_dataset():
    TrainRate = 0.6
    ValRate = 0.2
    OldPath = "/home/hp-video/Documents/zhangzhengyang/data/ADNI_JPG_NUM1"
    NewPath = "/home/hp-video/Documents/zhangzhengyang/data/ADNI_Dataset"
    _split_train_test(TrainRate, ValRate, OldPath, NewPath)

def _split_train_test2(train_rate, val_rate, old_path, new_path):
    dir_list = os.listdir(old_path)
    for dir_name in dir_list:
        dir_path = os.path.join(old_path, dir_name)
        image_list = os.listdir(dir_path)
        random.shuffle(image_list)
        image_num = len(image_list)-80
        train_num = round(image_num * train_rate)
        val_num = round(image_num * val_rate)

        train_list = image_list[0:image_num]
        # val_list = image_list[train_num:train_num+val_num]
        test_list = image_list[image_num:]

        os.makedirs(new_path + "/Train", exist_ok=True)
        # os.makedirs(new_path + "/Val", exist_ok=True)
        os.makedirs(new_path + "/Test", exist_ok=True)
        for name in train_list:
            shutil.copy(os.path.join(dir_path, name), f"{new_path}/Train/{dir_name}_{name}")
        # for name in val_list:
        #     shutil.copy(os.path.join(dir_path, name), f"{new_path}/Val/{dir_name}_{name}")
        for name in test_list:
            shutil.copy(os.path.join(dir_path, name), f"{new_path}/Test/{dir_name}_{name}")


def get_train_test_dataset2():
    TrainRate = 0.7
    ValRate = 0.3
    OldPath = "/home/hp-video/Documents/zhangzhengyang/data/ADNI_JPG_NUM1"
    NewPath = "/home/hp-video/Documents/zhangzhengyang/data/ADNI_Dataset2"
    _split_train_test2(TrainRate, ValRate, OldPath, NewPath)

if __name__ == '__main__':
    get_train_test_dataset2()
