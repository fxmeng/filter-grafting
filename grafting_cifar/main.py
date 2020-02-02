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
from models.resnet_cifar import *
from models.resnet_imagenet import *
from models.wrn import Network
from models.mobilenetv2 import MobileNetV2
from models.densenet import DenseNet121
from grafting import noise,inside,outside,grafting_filter,outsideFC
parser = argparse.ArgumentParser(description='PyTorch Grafting Training')
#basic setting
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--r', default=None, type = int)
parser.add_argument('--device', default='cuda', type = str)
parser.add_argument('--s', default='1', type = str)
parser.add_argument('--model', default='resnet32', type = str)
parser.add_argument('--cifar', default=10, type = int)
parser.add_argument('--print_frequence', default=100, type = int)
#Grafting setting
parser.add_argument('--sh', default='baseline', type = str,help='[entropy, norm, noise, inside, baseline]')
parser.add_argument('--threshold', default=0.1, type = float)
parser.add_argument('--num', default=100, type = int)
parser.add_argument('--i', default=1, type = int)
parser.add_argument('--w', default=500, type = int)
#Increase models diversity
parser.add_argument('--dynamiclr', action="store_true", default=False)
parser.add_argument('--difflr', action="store_true", default=False)
args = parser.parse_args()
print(args)
print('Session:%s\tModel:%d\tPID:%d'%(args.s,args.i,os.getpid()))
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.model=='resnet32':
    net = resnet32(args.cifar).to(args.device)
elif args.model=='resnet56':
    net = resnet56(args.cifar).to(args.device)
elif args.model=='resnet110':
    net = resnet110(args.cifar).to(args.device)
elif args.model=='mobilenetv2':
    net = MobileNetV2(args.cifar).to(args.device)
elif args.model=='densenet':
    net = DenseNet121(args.cifar).to(args.device)
elif args.model=='wrn':
    net = Network(args.cifar).to(args.device)

start_epoch = 0 
accuracy=[] 
if args.difflr:
    loc=(1+np.cos(np.pi*((args.num-args.i)/args.num)))/2
else:
    loc=1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr*loc, momentum=0.9, weight_decay=5e-4)
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
if args.cifar==10:
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
elif args.cifar==100:
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
if args.dynamiclr==True:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs*trainloader.__len__())
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=60,gamma=0.1)
def test(epoch):
    global accuracy
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    print('Session: %s    Network:%d    epoch:%d    accuracy:%.3f'%(args.s,args.i,epoch,acc))
    accuracy.append(acc) 
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx%args.print_frequence==args.print_frequence-1 or args.print_frequence==trainloader.__len__()-1:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if args.dynamiclr==True:
            lr_scheduler.step()
    if args.dynamiclr==False:
        lr_scheduler.step()

if args.r!=None:
    start_epoch = args.r+1
    checkpoint = torch.load('./checkpoint/%s/ckpt%d_%d.t7'%(args.s,args.i%args.num,args.r))
    net.load_state_dict(checkpoint['net']) 
    accuracy=checkpoint['acc'] 
    optimizer=checkpoint['optim'] 
    if args.sh=='entropy'or args.sh=='norm':
        outside(net,args.r)
    if args.dynamiclr==True:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs*trainloader.__len__(),last_epoch=args.r*trainloader.__len__())
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=60,gamma=0.1,last_epoch=args.r)

if __name__=='__main__':
    for epoch in range(start_epoch,args.epochs): 
        train(epoch)
        test(epoch)
        state = {
        'net': net.state_dict(),
        'optim':optimizer,
        'acc': accuracy
        }
        torch.save(state,'./checkpoint/%s/ckpt%d_%d.t7'%(args.s,args.i%args.num,epoch))
        if args.sh=='noise':
            noise(args,net,epoch)
        elif args.sh=='inside':
            inside(args,net,epoch)
        elif args.sh=='entropy'or args.sh=='norm':
            outside(args,net,epoch)
        elif args.sh=='grafting_filter':
            grafting_filter(args,net,epoch)
        elif args.sh=='FC':
            outsideFC(args,net,epoch)
        #else:baseline
    torch.save(accuracy,'./checkpoint/%s/accuracy%d.t7'%(args.s,args.i))

