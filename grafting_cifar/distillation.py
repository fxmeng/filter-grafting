#cifar100_wrn28-10
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import pickle
import numpy as np
import time
import collections
import math
from models.resnet import *
from models.mobilenetv2 import MobileNetV2
from models.wrn import Network
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('-e', '--epochs', default=200, type=int,
                    help='number of total epochs (default: 200)')
parser.add_argument('--save-dir', default='./checkpoint/', type=str,
                    help='directory of saved model (default: model/saved)')
parser.add_argument('--gpu', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--s', default='', type = str, 
                    help='Session')
parser.add_argument('--teacher', default='', type = str,)
parser.add_argument('--r', default=None, type = int, 
                    help='resume from checkpoint r')
parser.add_argument('--model', default='resnet20', type = str)
parser.add_argument('--cifar', default=100, type = int)
args = parser.parse_args()
print(args)
print('Session:%s\tPID:%d'%(args.s,os.getpid()))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if device == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
start_epoch=0
best_acc=0
model_list=[]
load_list=[args.teacher]
if args.model=='resnet20':
    for i,ckpt in enumerate(load_list):
        model=resnet20(args.cifar).to('cuda')
        model.load_state_dict(torch.load(ckpt)['net'])
        model.eval()
        model_list.append(model)
    net=resnet20(args.cifar).to('cuda')
elif args.model=='resnet32':
    for i,ckpt in enumerate(load_list):
        model=resnet32(args.cifar).to('cuda')
        model.load_state_dict(torch.load(ckpt)['net'])
        model.eval()
        model_list.append(model)
    net=resnet32(args.cifar).to('cuda')
elif args.model=='resnet32':
    for i,ckpt in enumerate(load_list):
        model=resnet32(args.cifar).to('cuda')
        model.load_state_dict(torch.load(ckpt)['net'])
        model.eval()
        model_list.append(model)
    net=resnet32(args.cifar).to('cuda')
if args.model=='resnet56':
    for i,ckpt in enumerate(load_list):
        model=resnet56(args.cifar).to('cuda')
        model.load_state_dict(torch.load(ckpt)['net'])
        model.eval()
        model_list.append(model)
    net=resnet56(args.cifar).to('cuda')
elif args.model=='resnet56':
    for i,ckpt in enumerate(load_list):
        model=resnet56(args.cifar).to('cuda')
        model.load_state_dict(torch.load(ckpt)['net'])
        model.eval()
        model_list.append(model)
    net=resnet56(args.cifar).to('cuda')
elif args.model=='resnet110':
    for i,ckpt in enumerate(load_list):
        model=resnet110(args.cifar).to('cuda')
        model.load_state_dict(torch.load(ckpt)['net'])
        model.eval()
        model_list.append(model)
    net=resnet110(args.cifar).to('cuda')
elif args.model=='wrn':
    for i,ckpt in enumerate(load_list):
        model=wrn(args.cifar).to('cuda')
        model.load_state_dict(torch.load(ckpt)['net'])
        model.eval()
        model_list.append(model)
    net=wrn(args.cifar).to('cuda')
elif args.model=='mobilenetv2':
    for i,ckpt in enumerate(load_list):
        model=MobileNetV2(args.cifar).to('cuda')
        model.load_state_dict(torch.load(ckpt)['net'])
        model.eval()
        model_list.append(model)
    net=MobileNetV2(args.cifar).to('cuda')

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
criterion = nn.CrossEntropyLoss()
if args.cifar==10:      
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.cifar==100:
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    # Save checkpoint.
    acc = 100.*correct/total
    print('Session: %s    epoch:%d    accuracy:%.3f'%(args.s,epoch,acc))
    if acc>best_acc:
        best_acc=acc
    state = {'net': net.state_dict(),
             'best_acc':acc}
    torch.save(state, './checkpoint/%s/ckpt_%d.t7'%(args.s,epoch))
T=2
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1**(epoch//60)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        teachers_outputs = torch.mean(torch.cat([model(inputs).unsqueeze(0) for model in model_list]), 0).squeeze()
        loss += (- F.log_softmax(outputs / T, 1) * F.softmax(teachers_outputs,1)).sum(dim = 1).mean() * T * T 
        loss.backward()
        optimizer.step()
for epoch in range(start_epoch,args.epochs):
    train(epoch)
    test(epoch)