import os
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Any, Callable, Optional, Tuple, List


class DatasetADNI(data.Dataset):
    def __init__(self, root, transforms: Optional[Callable] = None, ret_name = False):
        super(DatasetADNI, self).__init__()
        self.root = root
        self.name_list = os.listdir(root)
        self.name2id = {"AD": 0, "MCI": 1, "NC": 2}
        self.transforms = transforms
        self.CLASS_NAME = ("AD", "MCI", "NC")
        self.ret_name = ret_name
    def _load_image(self, name: str) -> np.array:
        image = Image.open(os.path.join(self.root, name))
        return np.expand_dims(np.array(image), axis=0)/255

    def _load_target(self, name: str) -> int:
        return self.name2id[name.split('_')[0]]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        name = self.name_list[index]
        image = self._load_image(name)
        target = self._load_target(name)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        image = F.pad(image, (3, 3, 21, 21))  # b 1 224 224
        # image = F.pad(image, (3, 3, 1, 1))
        if self.ret_name:
            return name, image, target
        else:
            return image, target

    def __len__(self) -> int:
        return len(self.name_list)

class DatasetADNI2(data.Dataset):
    def __init__(self, imgs, labels, transforms: Optional[Callable] = None, ret_name = False):
        super(DatasetADNI2, self).__init__()
        self.imgs_path = imgs
        self.labels = labels
        self.transforms = transforms
        
        self.ret_name = ret_name
    def _load_image(self, name: str) -> np.array:
        image = Image.open(name)
        return np.expand_dims(np.array(image), axis=0)/255

    def _load_target(self, name: str) -> int:
        return self.name2id[name.split('_')[0]]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        name = self.imgs_path[index]
        image = self._load_image(name)
        target = self.labels[index]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        image = F.pad(image, (3, 3, 21, 21))  # b 1 224 224
        # image = F.pad(image, (3, 3, 1, 1))
        if self.ret_name:
            return name, image, target
        else:
            return image, target

    def __len__(self) -> int:
        return len(self.imgs_path)

def collate_fn(data):
    imgs_list, targets_list = zip(*data)
    assert len(imgs_list) == len(targets_list)

    batch_imgs = torch.stack(imgs_list)
    batch_targets = torch.stack(targets_list)

    return batch_imgs, batch_targets


def collate_fn_name(data):
    name_list, imgs_list, targets_list = zip(*data)
    assert len(imgs_list) == len(targets_list)
    
    batch_names = name_list
    batch_imgs = torch.stack(imgs_list)
    batch_targets = torch.stack(targets_list)

    return batch_names, batch_imgs, batch_targets

class ToTensor(object):
    def __call__(self, image, target):
        image_tensor = torch.FloatTensor(image)
        target_temsor = torch.LongTensor([target])[0]
        return image_tensor, target_temsor


def get_dataset(batch_size,root = "/home/hp-video/Documents/zhangzhengyang/AD-GAT/ADNI_Dataset", num_workers=4):
    
    # root = "/home/hp-video/Documents/zhangzhengyang/data/ADNI_Dataset"
    train_dataset = DatasetADNI(os.path.join(root, "Train"), ToTensor())
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                       collate_fn=collate_fn)

    val_dataset = DatasetADNI(os.path.join(root, "Val"), ToTensor())
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     collate_fn=collate_fn)

    test_dataset = DatasetADNI(os.path.join(root, "cam"), ToTensor(),ret_name=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=collate_fn_name)

    return train_dataloader, val_dataloader, test_dataloader, len(train_dataset.CLASS_NAME)


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader, num_classes = get_dataset(4)
    for step, data in enumerate(val_dataloader):
        batch_imgs, batch_targets = data
        print(batch_imgs.shape, batch_targets)
    print('done')
