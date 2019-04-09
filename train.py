'''
ErfNet
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)
'''
import numpy as np
import torch

import torch.nn as nn
#import matplotlib.pyplot as plt

from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize,Scale,Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import train,test
#from yf import *
#from criterion import CrossEntropyLoss2d
from transform import Relabel, ToLabel, Colorize


import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import numpy as np

import random

import time
import shutil
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler

from erfnet import ERFNet

log_fmt = '%(levelname)s: %(message)s\t%(asctime)s'
formatter = logging.Formatter(log_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file_handler = TimedRotatingFileHandler(filename="./erf_save/loss")
#log_file_handler.suffix = "%Y-%m-%d_%H-%M.log"
log_file_handler.setFormatter(formatter)
logger.addHandler(log_file_handler)


NUM_CHANNELS = 1
NUM_CLASSES = 2

modelpth='./erf_save/model/'

color_transform = Colorize()
image_transform = ToPILImage()
input_transform = Compose([
    Resize((224,224)),
    #CenterCrop(256),
    ToTensor(),
    #Normalize([112.65,112.65,112.65],[32.43,32.43,32.43])
    #Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform = Compose([
    Resize((224,224)),
    #CenterCrop(324),
    ToLabel(),
    #Relabel(255, 1),
])

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):

        super(CrossEntropyLoss2d,self).__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

def train(model):

    num_epochs=20
    weight = torch.ones(NUM_CLASSES)
    weight[0] = 0

    loader = DataLoader(train(input_transform, target_transform),num_workers=1, batch_size=1, shuffle=True)
 

    weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)
    #base_lr=1e-3
    #weight_decay=0.0005
    #optimizer = Adam(model.parameters())
    #optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr }, {'params': get_10x_lr_params(model), 'lr': 10*base_lr} ], lr = base_lr, momentum = 0.9,weight_decay = weight_decay)
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)      ## scheduler 2
    lambda1 = lambda epoch: pow((1-((epoch-1)/num_epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2
    

    #optimizer = optim.SGD(model.parameters(), 1e-3, .9) 
    temp_loss=0
    #d=open('loss.txt','w')

    for epoch in range(num_epochs):
        epoch_loss = []
        temp_loss=0
        for step, (images, labels) in enumerate(loader):
            
            images = images.cuda()
            labels = labels.cuda()

            print labels.size()
            print images.size()
            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)
            
            optimizer.zero_grad()
            print outputs.size()
            print targets[:, 0].size()
     	   # loss = loss_calc(outputs[0], labels[0])
           # temp_loss+=float(loss)
            loss = criterion(outputs, targets[:, 0])

            loss.backward()
            optimizer.step()

            temp_loss+=float(loss.data[0])
           # epoch_loss.append(loss.data[0])

            if (step+1) % 500 == 0:
                #d.write(str(temp_loss/500)+'\n')
                logger.info(str(temp_loss/500))
                temp_loss=0
            if (step+2) % 6000 == 0:
		filename = 'main-'+'erfnet'+'-step-'+str(step)+'-epoch-'+str(epoch)+'.pth'
                torch.save(model.state_dict(), modelpth+filename)


def main():
    Net = ERFNet(NUM_CLASSES)
    Net = Net.cuda()
    train(Net)

if __name__ == '__main__':

    main()
