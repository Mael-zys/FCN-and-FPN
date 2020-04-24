import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import cv2
from onehot import onehot

#DataPath="/home/zhangyangsong/OCR/CTPNshare/data_ready/MSRA_TD500/train_im"
#MaskPath="/home/zhangyangsong/OCR/CTPNshare/data_ready/MSRA_TD500/train_mask"
DataPath="/home/zhangyangsong/OCR/ICDAR2015/train_im"
MaskPath="/home/zhangyangsong/OCR/ICDAR2015/train_mask"

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir(DataPath))

    def __getitem__(self, idx):
        img_name = os.listdir(DataPath)[idx]
        imgA = cv2.imread(DataPath+'/'+img_name)
        #imgA = cv2.resize(imgA, (160, 160))
        #img_name, _ = os.path.splitext(img_name)
        #imgB = cv2.imread(MaskPath+'/mask_'+img_name+'.png', 0)
        imgB = cv2.imread(MaskPath+'/mask_'+img_name, 0)
        #imgB = cv2.resize(imgB, (160, 160))
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2,0,1)
        imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)    

        return imgA, imgB

bag = BagDataset(transform)

train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

print('total data size is {}, train_size is {}, test size is {}'.format(len(bag),train_size,test_size))

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)


if __name__ =='__main__':

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)
