from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import visdom
import cv2
from test_data import test_dataloader
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
import os
import shutil


TEST_RESULT='test_result'
model_path='model_test/fcn_19.model'
# model_path='model/fcn_bb.model'
def test():
    #vis = visdom.Visdom()
    print(model_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg_model = VGGNet(pretrained=False,requires_grad=False, show_params=False)
    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=2)
    if not torch.cuda.is_available():
        fcn_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        fcn_model.load_state_dict(torch.load(model_path))
    #print(fcn_model)
    # fcn_model=torch.load(model_path)
    fcn_model = fcn_model.to(device)
    fcn_model.eval()
    miou=0
    num=0
    if os.path.exists(TEST_RESULT):
        shutil.rmtree(TEST_RESULT)

    os.mkdir(TEST_RESULT)

    for index, (bag, bag_msk) in enumerate(test_dataloader):
        with torch.no_grad():
            bag = bag.to(device)
            bag_msk = bag_msk.to(device)
            output = fcn_model(bag)
            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
            output_np = np.argmin(output_np, axis=1)
            output_np = np.squeeze(output_np[0, ...])
            bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
            bag_msk_np = np.argmin(bag_msk_np, axis=1)
            bag_msk_np = np.squeeze(bag_msk_np[0, ...])
            cv2.imwrite("test_result/"+str(index)+"_test.jpg",255*output_np)
            cv2.imwrite("test_result/"+str(index)+"_gt.jpg",255*bag_msk_np)
            inter=np.sum(np.multiply(output_np,bag_msk_np))
            union=np.sum(output_np)+np.sum(bag_msk_np)-inter
            miou+=inter/union
            num=index
    miou=miou/(num+1)
    print("MIOU is {}".format(miou))


if __name__ == "__main__":

    test()
    print(model_path)

