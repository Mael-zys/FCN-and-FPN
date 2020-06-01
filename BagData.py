import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import cv2
from onehot import onehot
import codecs
import random
from torchvision.transforms import transforms
from PIL import Image,ImageEnhance,ImageOps,ImageFile

MSRA = '/home/zhangyangsong/OCR/CTPNshare/data_ready/MSRA_TD500/train_im'  # MSRA_TD500数据集路径
ALI = '/home/zhangyangsong/OCR/CTPNshare/data_ready/ali_icpr/train_im' # ali_icpr数据集路径
ICDAR2015 = '/home/zhangyangsong/OCR/ICDAR2015/train_im'
ICDAR2013 = '/home/zhangyangsong/OCR/ICDAR2013/train_im'
MLT = '/home/zhangyangsong/OCR/MLT/train_im'
#ALI = '/home/zhangyangsong/OCR/CTPNshare/data_ready/MSRA_TD500'
DATASET_LIST = [MSRA,ICDAR2015,ICDAR2013,MLT]
#DATASET_LIST = [ICDAR2015]
#DataPath="/home/zhangyangsong/OCR/ICDAR2015/train_im"

def randomRotation(image,mask,nc,mode=Image.BICUBIC):
    '''mode:邻近插值，双线性插值，双三次样条插值（default）'''
    random_angle=np.random.randint(-20,20)
    return image.rotate(random_angle,mode), mask.rotate(random_angle), nc.rotate(random_angle)

cropsize_list = [256,288,320,352,384,416,448,480]
def randomcrop(image, mask):
    image_width=image.size[0]
    image_height=image.size[1]
    # crop_win_size=np.random.randint(256,480) #min,max为相对值
    crop_win_size = np.random.choice(cropsize_list,1)[0]
    random_region=((image_width-crop_win_size)>>1,(image_height-crop_win_size)>>1,(image_width+crop_win_size)>>1,(image_height+crop_win_size)>>1)
    # print(crop_win_size)
    # print(random_region)
    return image.crop(random_region), mask.crop(random_region)

def randomGaussian(image, mean=0.2, sigma=0.3):
    """
    对图像进行高斯噪声处理
    :param image:
    :return:
    """

    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        """
        对图像做高斯噪音处理
        :param im: 单通道图像
        :param mean: 偏移量
        :param sigma: 标准差
        :return:
        """
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    # 将图像转化成数组
    img = np.asarray(image)
    img = np.array(img)
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(0, 21) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 21) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度 



transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

resize_list = [512,576,640,704,800,960]
def scale_img(img, gt):
    height = img.shape[0]
    width = img.shape[1]
    min_size = np.random.choice(resize_list,1)[0]
    im_size_min=min(height,width)
    im_size_max=max(height,width)
    max_size = 1440
    step=16

    im_scale = float(min_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    new_h = int(height * im_scale)
    new_w = int(width * im_scale)

    new_h = new_h if new_h // step == 0 else (new_h // step + 1) * step
    new_w = new_w if new_w // step == 0 else (new_w // step + 1) * step

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 求出h和w方向上的缩放比例
    h_scale = float(img.shape[0])/float(height)
    w_scale = float(img.shape[1])/float(width)
    scale_gt = []
    for box in gt:
        scale_box = []
        for i in range(len(box)):
            scale_box.append([int(int(box[i][0]) * w_scale),int(int(box[i][1]) * h_scale)])
            
        scale_gt.append(scale_box)
    # 返回缩放后的图片和label
    return img, scale_gt


def read_gt_file(path, have_BOM=False):
    result = []
    not_care=[]
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
    return result, not_care

def shrink_mask(gt,h=0.7,w=0.9):
    h_scale = 1 - h
    w_scale = 1 - w
    scale_gt = []
    for box in gt:
        scale_box = [[[],[],[],[]]]
        # print(box)
        # print(box[0][0])
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
        

        root, file_name = os.path.split(img_name)  # 拆成路径和带后缀图片名
        root, _ = os.path.split(DataPath)  # root为图片所在文件夹
        name, _ = os.path.splitext(file_name)  # name为不带后缀文件名
        gt_name = 'gt_' + name + '.txt'  # 构造gt_name.txt
        gt_path = os.path.join(root, "train_gt", gt_name)  # root_train_gt_gt_name
        if not os.path.exists(gt_path):  # 如果图片对应的label不存在的话跳过
            print('Ground truth file of image {0} not exists.'.format(img_name))
            exit
            #imgB = cv2.imread(GtPath+'/gt_'+img_name, 0)
        gt_txt, not_care = read_gt_file(gt_path,True)  # 读取对应的标签，把标签转为list返回，每个元素为一个bbox的4个坐标 
        
        # imgA, gt_txt = scale_img(imgA, gt_txt)
        ima_shape=imgA.shape
        imgA = cv2.resize(imgA, (640, 640))
        

        gt_txt = shrink_mask(gt_txt,0.7,0.8)
        not_care = shrink_mask(not_care,0.7,0.8)

        b  = np.array(gt_txt, dtype = np.int32)
        imgB = np.zeros(ima_shape[:2], dtype = "uint8")
        cv2.polylines(imgB, b, 1, 255)
        cv2.fillPoly(imgB, b, 255)
        imgB = cv2.resize(imgB, (640, 640))
        
        c  = np.array(not_care, dtype = np.int32)
        imgC = np.zeros(ima_shape[:2], dtype = "uint8")
        cv2.polylines(imgC, c, 1, 255)
        cv2.fillPoly(imgC, c, 255)
        imgC = cv2.resize(imgC, (640, 640))

        imgA = Image.fromarray(np.uint8(imgA))
        imgB = Image.fromarray(np.uint8(imgB))
        imgC = Image.fromarray(np.uint8(imgC))
        choose = np.random.choice([True,False],4,p=[0.25,0.75])
        if choose[0] == True:    
            imgA = randomColor(imgA)
        if choose[1] == True:
            imgA = randomGaussian(imgA)
        # if choose[2] == True:    
        #     imgA, imgB = randomcrop(imgA, imgB)
        if choose[2] == True:
            imgA, imgB, imgC = randomRotation(imgA, imgB, imgC)
        imgA = np.array(imgA)
        imgB = np.array(imgB)
        imgC = np.array(imgC)
        # print(imgA.shape)
        # print(imgB.shape)
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
            #print(imgB.shape)
        imgB = imgB.transpose(2,0,1)
        imgB = torch.FloatTensor(imgB)


        imgC = imgC/255
        imgC = imgC.astype('uint8')
        # imgC = onehot(imgC, 2)
            #print(imgB.shape)
        
        # imgC = imgC.transpose(2,0,1)
        imgC = torch.FloatTensor(imgC)
        imgC = torch.unsqueeze(imgC,dim=0)
        # print(imgC.shape)
        if self.transform:
            imgA = self.transform(imgA)    
            #print(imgA.shape)
        return imgA, imgB, imgC

bag = BagDataset(transform)


# train_size = 5000
# val_size = 500
# other_size = len(bag)-train_size-val_size


train_size = int(0.9 * len(bag))
val_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, val_size])
# train_dataset, test_dataset, _ = random_split(bag, [train_size, val_size, other_size])
print('total data size is {}, train_size is {}, val_size is {}'.format(len(bag),train_size,val_size))

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)


if __name__ =='__main__':

    imgA=cv2.imread('/home/zhangyangsong/OCR/ICDAR2015/train_im/img_54.jpg')
    gt_txt, not_care = read_gt_file('/home/zhangyangsong/OCR/ICDAR2015/train_gt/gt_img_54.txt',True)  # 读取对应的标签，把标签转为list返回，每个元素为一个bbox的4个坐标 
    # imgA,gt_txt = scale_img(imgA,gt_txt)
    # print(gt_txt)
    gt_txt = shrink_mask(gt_txt,0.6,0.9)
    b  = np.array(gt_txt, dtype = np.int32)
    imgB = np.zeros(imgA.shape[:2], dtype = "uint8")
    # cv2.polylines(imgB, b, 1, 255)
    # cv2.fillPoly(imgB, b, 255)
    # imgA = Image.fromarray(np.uint8(imgA))
    # imgB = Image.fromarray(np.uint8(imgB))
    # # choose = np.random.choice([True,False],4)
    # # print(choose)
    # # if choose[0] == True:    
    # #     imgA = randomColor(imgA)
    # #     print('1')
    # # if choose[1] == True:
    # #     imgA = randomGaussian(imgA)
    # #     print('1')
    # # if choose[2] == True:    
    # #     imgA, imgB = randomcrop(imgA, imgB)
    # # if choose[3] == True:
    # #     imgA, imgB = randomRotation(imgA, imgB)
    # # imgA, imgB = randomRotation(imgA, imgB)
    # imgA, imgB = randomcrop(imgA, imgB)
    # imgA = np.array(imgA)
    # imgB = np.array(imgB)
    # cv2.imwrite('bag.jpg',imgA)
    # cv2.imwrite('bag_msk1.jpg',imgB)
    # for test_batch in test_dataloader:
    #     print(test_batch)
    not_care = shrink_mask(not_care,0.7,0.8)
        
    c  = np.array(not_care, dtype = np.int32)
    imgC = np.zeros(imgA.shape[:2], dtype = "uint8")
    cv2.polylines(imgC, c, 1, 255)
    cv2.fillPoly(imgC, c, 255)
    imgC = cv2.resize(imgC, (640, 640))


    imgC = imgC/255
    imgC = imgC.astype('uint8')
        # imgC = onehot(imgC, 2)
            #print(imgB.shape)
        
        # imgC = imgC.transpose(2,0,1)
    imgC = torch.FloatTensor(imgC)
    imgC = torch.unsqueeze(imgC,dim=0)
    print(imgC.shape)
