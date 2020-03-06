from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import collections
import time
import argparse
import sys
import pickle
import numpy as np
import time
from models.resnet import *
from models.resnet_leaky import *
from models.mobilenetv2 import MobileNetV2

parser = argparse.ArgumentParser(description='PyTorch Grafting Training')
# basic setting
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--r', default=None, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--s', default='1', type=str)
parser.add_argument('--model', default='resnet32', type=str)
parser.add_argument('--cifar', default=10, type=int)
parser.add_argument('--print_frequence', default=1000, type=int)
# Grafting setting
parser.add_argument('--a', default=0.4, type=float)
parser.add_argument('--c', default=500, type=int)
parser.add_argument('--num', default=100, type=int)
parser.add_argument('--i', default=1, type=int)
# Increase models diversity
parser.add_argument('--cos', action="store_true", default=False)
parser.add_argument('--difflr', action="store_true", default=False)
args = parser.parse_args()
print(args)
print('Session:%s\tModel:%d\tPID:%d' % (args.s, args.i, os.getpid()))
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.model == 'resnet20':
    net = resnet20(args.cifar).to(args.device)
elif args.model == 'resnet32':
    net = resnet32(args.cifar).to(args.device)
elif args.model == 'resnet56':
    net = resnet56(args.cifar).to(args.device)
elif args.model == 'resnet110':
    net = resnet110(args.cifar).to(args.device)
elif args.model == 'resnet20_leaky':
    net = resnet20_leaky(args.cifar).to(args.device)
elif args.model == 'resnet32_leaky':
    net = resnet32_leaky(args.cifar).to(args.device)
elif args.model == 'resnet56_leaky':
    net = resnet56_leaky(args.cifar).to(args.device)
elif args.model == 'resnet110_leaky':
    net = resnet110_leaky(args.cifar).to(args.device)
elif args.model == 'mobilenetv2':
    net = MobileNetV2(args.cifar).to(args.device)
start_epoch = 0
best_acc = 0
if args.difflr:
    loc = (1 + np.cos(np.pi * ((args.num - args.i) / args.num))) / 2
else:
    loc = 1
print('The initial learning rate is:', args.lr * loc)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr * loc, momentum=0.9, weight_decay=5e-4)
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
if args.cifar == 10:
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
elif args.cifar == 100:
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
if args.cos == True:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * trainloader.__len__())
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)


def entropy(x, n=10):
    x = x.reshape(-1)
    scale = (x.max() - x.min()) / n
    entropy = 0
    for i in range(n):
        p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
        if p != 0:
            entropy -= p * torch.log(p)
    return float(entropy.cpu())


def grafting(net, epoch):
    while True:
        try:
            checkpoint = torch.load('%s/ckpt%d_%d.t7' % (args.s, args.i - 1, epoch))['net']
            break
        except:
            time.sleep(10)
    model = collections.OrderedDict()
    for i, (key, u) in enumerate(net.state_dict().items()):
        if 'conv' in key:
            w = round(args.a / np.pi * np.arctan(args.c * (entropy(u) - entropy(checkpoint[key]))) + 0.5, 2)
        model[key] = u * w + checkpoint[key] * (1 - w)
    net.load_state_dict(model)


def test(epoch):
    global best_acc
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
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        state = {
            'net': net.state_dict(),
            'acc': acc
        }
        torch.save(state, '%s/best_%d.t7' % (args.s, epoch))
    print('Network:%d    epoch:%d    accuracy:%.3f    best:%.3f' % (args.i, epoch, acc, best_acc))


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
        if batch_idx % args.print_frequence == args.print_frequence - 1 or args.print_frequence == trainloader.__len__() - 1:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if args.cos == True:
            lr_scheduler.step()
    if args.cos == False:
        lr_scheduler.step()


if __name__ == '__main__':
    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        test(epoch)
        state = {
            'net': net.state_dict(),
        }
        torch.save(state, '%s/ckpt%d_%d.t7' % (args.s, args.i % args.num, epoch))
        grafting(net, epoch)
