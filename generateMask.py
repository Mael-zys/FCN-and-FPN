# coding=utf-8
import numpy as np

import cv2
import matplotlib.pyplot as plt
import os  # 输入输出库
import codecs

ALI = '/home/zhangyangsong/OCR/ICDAR2015'
#ALI = '/home/zhangyangsong/OCR/CTPNshare/data_ready/MSRA_TD500'
# DATASET_LIST = [MSRA, ALI]
DATASET_LIST = [ALI]

def loop_files(path):
    files = []
    l = os.listdir(path)
    for f in l:
        files.append(os.path.join(path, f))
    return files


def create_train_val():
    train_im_list = []
    test_im_list = []
    train_gt_list = []
    test_gt_list = []
    for dataset in DATASET_LIST:
        trains_im_path =os.path.join(dataset, 'train_im')
        tests_im_path = os.path.join(dataset, 'test_im')
        trains_gt_path =os.path.join(dataset, 'train_gt')
        test_gt_path = os.path.join(dataset, 'test_gt')
        train_im = loop_files(trains_im_path)
        train_gt = loop_files(trains_gt_path)
        test_im = loop_files(tests_im_path)
        test_gt = loop_files(test_gt_path)
        train_im_list += train_im
        test_im_list += test_im
        train_gt_list += train_gt
        test_gt_list += test_gt
    return train_im_list, train_gt_list, test_im_list, test_gt_list

def read_gt_file(path, have_BOM=False):
    result = []
    if have_BOM:
        fp = codecs.open(path, 'r', 'utf-8-sig')
    else:
        fp = open(path, 'r')
    for line in fp.readlines():
        pt = line.split(',')
        if have_BOM:
            box = [[int(round(float(pt[0]))),int(round(float(pt[1])))],[int(round(float(pt[2]))),int(round(float(pt[3])))],[int(round(float(pt[4]))),int(round(float(pt[5])))],[int(round(float(pt[6]))),int(round(float(pt[7])))]]
        else:
            box = [[int(round(float(pt[0]))),int(round(float(pt[1])))],[int(round(float(pt[2]))),int(round(float(pt[3])))],[int(round(float(pt[4]))),int(round(float(pt[5])))],[int(round(float(pt[6]))),int(round(float(pt[7])))]]
        result.append(box)
    fp.close()
    return result

if __name__ == '__main__':
    train_im_list, train_gt_list, val_im_list, val_gt_list = create_train_val()
    
    for im in train_im_list: # 每次只训练一张照片么？？？
        root, file_name = os.path.split(im)  # 拆成路径和带后缀图片名
        root, _ = os.path.split(root)  # root为图片所在文件夹
        name, _ = os.path.splitext(file_name)  # name为不带后缀文件名
        gt_name = 'gt_' + name + '.txt'  # 构造gt_name.txt

        gt_path = os.path.join(root, "train_gt", gt_name)  # root_train_gt_gt_name

        if not os.path.exists(gt_path):  # 如果图片对应的label不存在的话跳过
            print('Ground truth file of image {0} not exists.'.format(im))
            continue

        gt_txt = read_gt_file(gt_path,True)  # 读取对应的标签，把标签转为list返回，每个元素为一个bbox的4个坐标
        image = cv2.imread(im) # 把图片读取为opencv对象  
    
        b  = np.array(gt_txt, dtype = np.int32)
        im = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.polylines(im, b, 1, 255)
        cv2.fillPoly(im, b, 255)

        mask = im
        mask_name = 'mask_' + name + '.jpg'
        maskpath=os.path.join(root, "train_mask", mask_name)
        cv2.imwrite(maskpath,mask)


    for im in val_im_list: # 每次只训练一张照片么？？？
        root, file_name = os.path.split(im)  # 拆成路径和带后缀图片名
        root, _ = os.path.split(root)  # root为图片所在文件夹
        name, _ = os.path.splitext(file_name)  # name为不带后缀文件名
        gt_name = 'gt_' + name + '.txt'  # 构造gt_name.txt

        gt_path = os.path.join(root, "test_gt", gt_name)  # root_train_gt_gt_name

        if not os.path.exists(gt_path):  # 如果图片对应的label不存在的话跳过
            print('Ground truth file of image {0} not exists.'.format(im))
            continue

        gt_txt = read_gt_file(gt_path,True)  # 读取对应的标签，把标签转为list返回，每个元素为一个bbox的4个坐标
        image = cv2.imread(im) # 把图片读取为opencv对象
        b  = np.array(gt_txt, dtype = np.int32)

        im = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.polylines(im, b, 1, 255)
        cv2.fillPoly(im, b, 255)

        mask = im
        mask_name = 'mask_' + name + '.jpg'
        maskpath=os.path.join(root, "test_mask", mask_name)
        cv2.imwrite(maskpath,mask)