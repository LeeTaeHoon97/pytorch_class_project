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
import os


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

      

class Model():
    def __init__(self):
        device=torch.device('cpu')
        self.path = "C:/dev/Pytorch-workspace_class/teamproject"
        self.model=Net()
        self.model.load_state_dict(torch.load(self.path+'\emotion_recognition.pt',map_location=device))
        self.model.eval()

        self.simple_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.Grayscale(), transforms.ToTensor(),
             transforms.Normalize([0.485], [0.229])])

    def get_value(self):
        valid = ImageFolder(self.path+"/images", self.simple_transform)
        validloader = DataLoader(valid)
        dataiter = iter(validloader)
        images, labels = dataiter.next()

        # 이미지를 넘기고, 이미지 삭제
        toList = torch.exp(self.model(images)).tolist()
        toRoundedList = [[i * 100 for i in nest] for nest in toList]
        toRoundedList = [[round(num, 1) for num in nest] for nest in toRoundedList]
        

        return toRoundedList  # 이값을 리턴

