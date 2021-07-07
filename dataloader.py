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