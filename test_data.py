import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import cv2
from onehot import onehot
import codecs

MSRA = '/home/zhangyangsong/OCR/CTPNshare/data_ready/MSRA_TD500/test_im'  # MSRA_TD500数据集路径
ALI = '/home/zhangyangsong/OCR/CTPNshare/data_ready/ali_icpr/test_im' # ali_icpr数据集路径
ICDAR2015 = '/home/zhangyangsong/OCR/ICDAR2015/test_im'
ICDAR2013 = '/home/zhangyangsong/OCR/ICDAR2013/test_im'
MLT = '/home/zhangyangsong/OCR/MLT/test_im'
#ALI = '/home/zhangyangsong/OCR/CTPNshare/data_ready/MSRA_TD500'
#DATASET_LIST = [MSRA,ICDAR2015,ICDAR2013,MLT]
DATASET_LIST = [ICDAR2015]
#DataPath="/home/zhangyangsong/OCR/ICDAR2015/train_im"

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def shrink_mask(gt,h=0.7,w=0.9):
    # height = img.shape[0]
    # width = img.shape[1]
    # 求出h和w方向上的缩放比例
    # h_scale = 0.6
    # w_scale = 0.9
    h_scale = 1 - h
    w_scale = 1 - w
    scale_gt = []
    for box in gt:
        scale_box = [[[],[],[],[]]]
        scale_box[0][0].append(int(round(float(box[0][0])+w_scale/2*(box[1][0]-box[0][0]))))
        scale_box[0][0].append(int(round(float(box[0][1])+h_scale/2*(box[3][1]-box[0][1]))))
        scale_box[0][1].append(int(round(float(box[1][0])-w_scale/2*(box[1][0]-box[0][0]))))
        scale_box[0][1].append(int(round(float(box[1][1])+h_scale/2*(box[2][1]-box[1][1]))))
        scale_box[0][2].append(int(round(float(box[2][0])-w_scale/2*(box[2][0]-box[3][0]))))
        scale_box[0][2].append(int(round(float(box[2][1])-h_scale/2*(box[2][1]-box[1][1]))))
        scale_box[0][3].append(int(round(float(box[3][0])+w_scale/2*(box[2][0]-box[3][0]))))
        scale_box[0][3].append(int(round(float(box[3][1])-h_scale/2*(box[3][1]-box[0][1]))))
        scale_gt.append(scale_box)
    # 返回缩放后的图片和label
    return scale_gt



def read_gt_file(path, have_BOM=False):
    result = []
    not_care = []
    if have_BOM:
        fp = codecs.open(path, 'r', 'utf-8-sig')
    else:
        fp = open(path, 'r')
    for line in fp.readlines():
        pt = line.split(',')
        if (len(pt)>8) and pt[8]=='###\r\n':
            nc = [[int(round(float(pt[0]))),int(round(float(pt[1])))],[int(round(float(pt[2]))),int(round(float(pt[3])))],[int(round(float(pt[4]))),int(round(float(pt[5])))],[int(round(float(pt[6]))),int(round(float(pt[7])))]]
            not_care.append(nc)
            continue
        if have_BOM:
            box = [[int(round(float(pt[0]))),int(round(float(pt[1])))],[int(round(float(pt[2]))),int(round(float(pt[3])))],[int(round(float(pt[4]))),int(round(float(pt[5])))],[int(round(float(pt[6]))),int(round(float(pt[7])))]]
        else:
            box = [[int(round(float(pt[0]))),int(round(float(pt[1])))],[int(round(float(pt[2]))),int(round(float(pt[3])))],[int(round(float(pt[4]))),int(round(float(pt[5])))],[int(round(float(pt[6]))),int(round(float(pt[7])))]]
        result.append(box)
    fp.close()
    return result,not_care

def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(1024) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1600:
        im_scale = float(1600) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 32 == 0 else (new_h // 32 + 1) * 32
    new_w = new_w if new_w // 32 == 0 else (new_w // 32 + 1) * 32

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])

class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        length=0
        for DataPath in DATASET_LIST :
            length+=len(os.listdir(DataPath))
        return length

    def __getitem__(self, idx):
        length=[]
        for DataPath in DATASET_LIST :
            length.append(len(os.listdir(DataPath)))
        i=0
        while idx >= length[i]:
            idx=idx % length[i]
            i+=1
        DataPath=DATASET_LIST[i]
        img_name = os.listdir(DataPath)[idx]
        imgA = cv2.imread(DataPath+'/'+img_name)
        #print('image {0} not exists.'.format(img_name))
        ima_shape=imgA.shape

        sizes = 800
        sizes1 = 480
        imgA = cv2.resize(imgA, (sizes, sizes))

        imgA1 = cv2.resize(imgA, (sizes1, sizes1))

        root, file_name = os.path.split(img_name)  # 拆成路径和带后缀图片名
        root, _ = os.path.split(DataPath)  # root为图片所在文件夹
        name, _ = os.path.splitext(file_name)  # name为不带后缀文件名
        gt_name = 'gt_' + name + '.txt'  # 构造gt_name.txt
        gt_path = os.path.join(root, "test_gt", gt_name)  # root_train_gt_gt_name
        if not os.path.exists(gt_path):  # 如果图片对应的label不存在的话跳过
            print('Ground truth file of image {0} not exists.'.format(img_name))
            exit
            #imgB = cv2.imread(GtPath+'/gt_'+img_name, 0)
        gt_txt, not_care = read_gt_file(gt_path,True)  # 读取对应的标签，把标签转为list返回，每个元素为一个bbox的4个坐标 
        gt_txt = shrink_mask(gt_txt,0.7,0.8)
        not_care = shrink_mask(not_care,0.7,0.8)
        
        b  = np.array(gt_txt, dtype = np.int32)
        imgB = np.zeros(ima_shape[:2], dtype = "uint8")
        cv2.polylines(imgB, b, 1, 255)
        cv2.fillPoly(imgB, b, 255)

        imgB = cv2.resize(imgB, (sizes, sizes))
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
            #print(imgB.shape)
        imgB = imgB.transpose(2,0,1)
        cv2.imwrite('aaa.jpg',imgB[0]*255)
        imgB = torch.FloatTensor(imgB)
        
        c  = np.array(not_care, dtype = np.int32)
        imgC = np.zeros(ima_shape[:2], dtype = "uint8")
        cv2.polylines(imgC, c, 1, 255)
        cv2.fillPoly(imgC, c, 255)

        imgC = cv2.resize(imgC, (sizes, sizes))
        imgC = imgC/255
        imgC = imgC.astype('uint8')
        # imgC = onehot(imgC, 2)
        #     #print(imgB.shape)
        # imgC = imgC.transpose(2,0,1)
        imgC = torch.FloatTensor(imgC)
        imgC = torch.unsqueeze(imgC,dim=0)
        if self.transform:
            imgA = self.transform(imgA)  
            imgA1 = self.transform(imgA1)  
            #print(imgA.shape)
        return imgA, imgA1, imgB, imgC, name, ima_shape

bag = BagDataset(transform)

# train_size = int(0.9 * len(bag))
# test_size = len(bag) - train_size
# train_dataset, test_dataset = random_split(bag, [train_size, test_size])

print('total test size is {}'.format(len(bag)))

test_dataloader = DataLoader(bag, batch_size=1, shuffle=False, num_workers=1)
#test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)


if __name__ =='__main__':

    for test_batch in test_dataloader:
        print(test_batch)
