import os
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import random
import torch

class SelfSupervisedData(Dataset):
    def __init__(self, img_path):
        super(SelfSupervisedData, self).__init__()
        self.img_path = img_path
        self.img_list = listdir(img_path)
        self.original = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])
        self.rotate180 = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            self.rot90_fn,
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                         ])
        self.grayscale = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.Grayscale(3),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                         ])
        self.randomcrop = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.RandomCrop((56, 56)),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          ])
        self.augments = [self.original, self.rotate180, self.grayscale, self.randomcrop]

    def rot90_fn(self, x):
        return torch.rot90(x, 2, [1, 2])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(join(self.img_path, self.img_list[index]))
        augment = random.choice([i for i in range(4)])
        img = self.augments[augment](img)
        return img, augment

class SupervisedData(Dataset):
    def __init__(self, img_path):
        super(SupervisedData, self).__init__()
        self.img_path = img_path
        self.img_list = listdir(img_path)
        self.transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                         ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(join(self.img_path, self.img_list[index]))
        img = self.transform(img)
        label = int(self.img_list[index][0:self.img_list[index].find('_')])
        return img, label

if __name__ == '__main__':
    dataset = SupervisedData('./dataset/data/p1_data/train_50/')
    loader = DataLoader(dataset, batch_size = 2, shuffle = True)
    for img, label in tqdm(loader):
        print(img.shape)
        print(label)
        break