#!/usr/bin/env python
# coding: utf-8


from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset,DataLoader
import time
from PIL import Image

import cascade



# if torch.cuda.is_available():
#   device_count = torch.cuda.device_count()
#   print("device_count: {}".format(device_count))
#   for device_num in range(device_count):
#     print("device {} capability {}".format(
#         device_num, torch.cuda.get_device_capability(device_num)))
#     print("device {} name {}".format(
#         device_num, torch.cuda.get_device_name(device_num)))
# else:
#   print("no cuda device")




# is_cuda=False
# if torch.cuda.is_available():
#     is_cuda = True
# print("is_cuda:",is_cuda)



#폴더안에 모든 파일 목록을 읽음
path= "C:\dev\Pytorch-workspace_class"   

simple_transform = transforms.Compose([transforms.Resize((224,224)),transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize([0.485], [0.229])])
   

##TJ_mk6용 net
#메모리 부족으로인한 배치사이즈 4변경 에폭 15 기준 적합, 이후 과적합 단 학습률 보장을위해 20회까지 돌림, 드롭아웃 재설정
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,stride=1,padding=0)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=0)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3,stride=1,padding=0)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3,stride=1,padding=1)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3,stride=1,padding=0)

        self.fc1 = nn.Linear(32*2*2, 256)
        
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)


    def forward(self, x):
        x= F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,2)
        x=F.relu(self.conv3(x))
        
        x=F.max_pool2d(x,2)
        x=F.relu(self.conv4(x))
        
        x=F.max_pool2d(x,2)
        x=F.relu(self.conv5(x))
        
        x=F.max_pool2d(x,2,padding=1)
        
        x=F.relu(self.conv6(x))
        x=F.max_pool2d(x,2,padding=1)

        x=F.relu(self.conv7(x))
        x=F.max_pool2d(x,2,padding=0)
        
        x=x.view(-1,32*2*2)
        x=F.relu(self.fc1(x))
        x = F.dropout(x,p=0.5, training=self.training)
        x=F.relu(self.fc2(x))
        x = F.dropout(x,p=0.5, training=self.training)
        x=self.fc3(x)

        return F.log_softmax(x,dim=1)

      


device=torch.device('cpu')
model=Net()
model.load_state_dict(torch.load(path+'\TJ_hoon_mk6.pt',map_location=device))
model.eval()




# data=ImageFolder("C:\dev\Pytorch-workspace_class\images", simple_transform)


is cut = True # 이부분 변수 받아서 수정.
#만약 조건문이 참이면, 캐스케이드를 해야 하면, 이미지 파일명이 뭔지를 알아야됨
if is_cut==False:      #이미지 안잘랐을때, 즉 원본
    cascade.do_cascade("C:/dev/Pytorch-workspace_class/team_project_1/backend/assets/python/img/이미지.jpg") #하고자하는 이미지  #웹 backend의 img폴더 위치 넣기
    valid = ImageFolder(path,simple_transform)
else:                  #잘린 이미지를 넘길때
    valid=ImageFolder("C:/dev/Pytorch-workspace_class/team_project_1/backend/assets/python/mod_img",simple_transform)    

# cascade.do_cascade("C:/dev/Pytorch-workspace_class/images/123.jpg")

validloader=DataLoader(valid)
dataiter=iter(validloader)
images,labels=dataiter.next()





#이미지를 넘기고, 이미지 삭제
toList = torch.exp(model(images)).tolist()
toRoundedList=[[i*100 for i in nest]for nest in toList]
toRoundedList = [[round(num,1) for num in nest]for nest in toRoundedList]


print(toRoundedList)        #이값을 리턴


# 이부분부터 리턴해줘야됨