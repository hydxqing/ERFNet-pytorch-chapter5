'''
ErfNet
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)
'''
import numpy as np
import torch

import os 
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.pylab import array

from PIL import Image
from argparse import ArgumentParser

from erfnet import ERFNet
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torchvision.transforms import Compose, CenterCrop, Normalize,Scale
from torchvision.transforms import ToTensor, ToPILImage

from transform import Relabel, ToLabel, Colorize

import time
import cv2
EXTENSIONS = ['.jpg', '.png']

NUM_CHANNELS = 1
NUM_CLASSES = 2

color_transform = Colorize()
input_transform = Compose([
    Scale((224,224)),
    ToTensor(),
])
target_transform = Compose([
    Scale((224,224)),
    ToLabel(),
])

def infer(model):
    print 'ok'
    label_np=array(Image.open('./data/test/Labels/10.png'))
    #label_np = cv2.cvtColor(label_np, cv2.COLOR_BGR2GRAY)
    img=Image.open('./data/test/Images/10.png')
    img_n=array(img)
    
    img_np=np.resize(img_n,(224,224,3))
    #image = f.convert('RGB')
    #img_cuda= input_transform(image).cuda()
    print img_np.shape 
    outputs = model(Variable(torch.from_numpy(img_np[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda())
    interp = nn.UpsamplingBilinear2d(size=(224, 224))
    outputs = interp(outputs[3]).cpu().data[0].numpy()
    outputs = outputs[:,:img_np.shape[0],:img_np.shape[1]]
    print outputs.shape
#    outputs = np.array(outputs)
#    outputs = torch.from_numpy(outputs)
#    print outputs.shape
   
    outputs = outputs.transpose(1,2,0)
    outputs = np.argmax(outputs,axis = 2)
    print outputs.shape
    mask=np.array(outputs)

    print type(mask)
    #sa=Image.fromarray(mask)
    #sa.save('1.png')
    cv2.imwrite('tuandao_10.png',mask*120)
    #mask=array(outputs).argmax(axis=0)
    print('mask',mask.shape)
    print mask.max()
    print(label_np.shape)
    #a=(mask==label_np)*1
    #same=a.sum()
    #acc=same/250000.0
    #print acc
#    acc.dtype='float32'
#    print acc/(500*500)
    #print(label_np.max())

    fig,ax=plt.subplots(1,3)
    ax[0].imshow(img_np,cmap='gray')
    ax[1].imshow(mask)
    ax[2].imshow(label_np)
    plt.show()
    return 0

   
def midfunc():
    print 'load Model....\n'
    t1=time.time()
    Net = ERFNet(2)
    model='./erf_save/model/main-erfnet-step-5998-epoch-20.pth'

    #print list(model.children())
    Net.load_state_dict(torch.load(model))
    
    Net = Net.cuda()
    print 'Done.\n'
    print 'time:',time.time()-t1,'\n'
    print 'compute output....\n'
    infer(Net)
    return 0

if __name__ =='__main__':

    midfunc()
