from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import visdom
import cv2
from test_data import test_dataloader
from FPN import FPN
import os
import shutil


TEST_RESULT='test_result_fpn'
model_path='model_fpn/fcn_9.model'
# model_path='model/fcn_bb.model'
def test():
    #vis = visdom.Visdom()
    # print(model_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcn_model = FPN([2,4,23,3], 2, back_bone="resnet")
    if not torch.cuda.is_available():
        fcn_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        fcn_model.load_state_dict(torch.load(model_path))
    # print(fcn_model)
    # fcn_model=torch.load(model_path)
    fcn_model = fcn_model.to(device)
    fcn_model.eval()
    miou=0
    num=0
    if os.path.exists(TEST_RESULT):
        shutil.rmtree(TEST_RESULT)

    os.mkdir(TEST_RESULT)

    for index, (bag, bag1, bag_msk, not_care, name, shape) in enumerate(test_dataloader):
        with torch.no_grad():
            bag = bag.to(device)
            bag1 = bag1.to(device)
            bag_msk = bag_msk.to(device)
            output = fcn_model(bag)
            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
            output_np = output_np[0,0,:,:]
            
            output1 = fcn_model(bag1)
            output_np1 = output1.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
            output_np1 = output_np1[0,0,:,:]
            output_np1 = cv2.resize(output_np1,(output_np.shape[1],output_np.shape[0]))

            output2 = np.zeros((2,output_np.shape[0],output_np.shape[1]))
            output2[0,:,:] = output_np
            output2[1,:,:] = output_np
            # output_np = np.append(output_np,output_np1,axis=0)
            # print(output2.shape)
            output_np = np.max(output2,axis=0)

            # output_np = np.argmin(output_np, axis=1)
            # output_np = np.squeeze(output_np[0, ...])
            bag_msk_np = bag_msk.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
            bag_msk_np = np.argmin(bag_msk_np, axis=1)
            bag_msk_np = np.squeeze(bag_msk_np[0, ...])
            
            # output_np = output_np[0,0,:,:]
            ind = np.where(output_np > 0.5)
            output_np = np.zeros(output_np.shape)
            output_np[ind]=1
            
            # print(name)
            # print(type(bag_msk_np))
            cv2.imwrite("test_result_fpn/"+name[0]+"_test.jpg",255*output_np)
            cv2.imwrite("test_result_fpn/"+name[0]+"_gt.jpg",255*bag_msk_np)
            
            not_care = not_care.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
            # not_care = np.argmin(not_care, axis=1)
            # print(not_care.shape)
            not_care=np.squeeze(not_care[0, ...])
            # not_care=np.squeeze(not_care[0, ...])
            # print(not_care.shape)
            ind = np.where(not_care==1)
            output_np[ind]=0

            inter=np.sum(np.multiply(output_np,bag_msk_np))
            union=np.sum(output_np)+np.sum(bag_msk_np)-inter
            # print(inter)
            # print(union)
            if union==0:
                continue
            miou+=inter/union
            num=index
    miou=miou/(num+1)
    print("MIOU is {}".format(miou))


if __name__ == "__main__":

    test()
    print(model_path)

