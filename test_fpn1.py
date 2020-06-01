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
submit_path = './submit'
# model_path='model/fcn_bb.model'
def test():
    #vis = visdom.Visdom()
    # print(model_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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

    if os.path.exists(TEST_RESULT):
        shutil.rmtree(TEST_RESULT)

    os.mkdir(TEST_RESULT)
    
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)

    os.mkdir(submit_path)
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
            output_np = np.zeros(output_np.shape,dtype=np.uint8)
            output_np[ind]=255
            
            output_np = cv2.resize(output_np,(shape[1],shape[0]))
            # img = cv2.cvtColor(output_np, cv2.COLOR_GRAY2RGB)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(output_np, 230, 255, cv2.THRESH_BINARY_INV)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = []
            for i,c in enumerate(contours):
                if i==0:
                    continue
                # 找面积最小的矩形
                area = cv2.contourArea(c)
                # print(area)
                if area < 20.0 :
                    continue
                rect = cv2.minAreaRect(c)
                # 得到最小矩形的坐标
                bbox = cv2.boxPoints(rect)
                bbox = bbox.astype('int32')
                bboxes.append(bbox.reshape(-1))
                # print('1')
            
            bag_msk_np = 255*bag_msk_np
            bag_msk_np = np.array(bag_msk_np,dtype = "uint8")
            for bbox in bboxes:
                # cv2.drawContours(bag_msk_np, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)
                cv2.polylines(bag_msk_np, [bbox.reshape(4, 2)], 1, 120)
                # cv2.fillPoly(bag_msk_np, [bbox.reshape(4, 2)], 120)
                # print(bbox.reshape(4, 2))
            seq = []
            if bboxes is not None:
                seq.extend([','.join([str(int(b)) for b in box]) + '\n' for box in bboxes])
            with open(os.path.join(submit_path, 'res_' + os.path.basename(name[0])+'.txt'), 'w') as f:
                f.writelines(seq)
            # print(name)
            # print(type(bag_msk_np))
            # cv2.imwrite("test_result_fpn/"+name[0]+"_test.jpg",255*output_np)
            # cv2.imwrite("test_result_fpn/"+name[0]+"_gt.jpg",bag_msk_np)
            



if __name__ == "__main__":

    test()
    print(model_path)

