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
from models.vgg import VGG

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
parser.add_argument('--level', type=str, default='filter')
args = parser.parse_args()
print(args)
print('Session:%s\tModel:%d\tPID:%d' % (args.s, args.i, os.getpid()))
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = VGG(deep=16, n_classes=args.cifar).to(args.device)
last_net = VGG(deep=16, n_classes=args.cifar).to(args.device)
start_epoch = 0
accuracy = []
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
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
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)


def BN(epoch):
    while not os.path.exists('%s/ckpt%d_%d.t7' % (args.s, args.i - 1, epoch)):
        time.sleep(10)
    while True:
        try:
            checkpoint = torch.load('%s/ckpt%d_%d.t7' % (args.s, args.i - 1, epoch))
            last_net.load_state_dict(checkpoint['net'])
            break
        except:
            time.sleep(10)
    bn = []
    for m1, m2 in zip(net.modules(), last_net.modules()):
        if isinstance(m1, nn.BatchNorm2d):
            #bn.append(torch.atan(500 * (m1.weight.data - m2.weight.data)) / np.pi + 1 / 2)
            bn.append(m1.weight.data.abs() / (m1.weight.data.abs() + m2.weight.data.abs()))
            for j, w in enumerate(bn[-1]):
                m1.weight.data = m1.weight.data.clone() * w + m2.weight.data.clone() * (1 - w)
                m1.bias.data = m1.bias.data.clone() * w + m2.bias.data.clone() * (1 - w)
                m1.running_var.data = m1.running_var.data.clone() * w + m2.running_var.data.clone() * (1 - w)
                m1.running_mean.data = m1.running_mean.data.clone() * w + m2.running_mean.data.clone() * (1 - w)
    i = -1
    for m1, m2 in zip(net.modules(), last_net.modules()):
        if isinstance(m1, nn.Conv2d):
            i += 1
            if i != 0:
                for j, w in enumerate(bn[i - 1]):
                    m1.weight.data[:, j] = m1.weight.data.clone()[:, j] * w + m2.weight.data.clone()[:, j] * (1 - w)
            for j, w in enumerate(bn[i]):
                m1.weight.data[j] = m1.weight.data.clone()[j] * w + m2.weight.data.clone()[j] * (1 - w)


def entropy(x, n=10):
    x = x.reshape(-1)
    scale = (x.max() - x.min()) / n
    entropy = 0
    for i in range(n):
        p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
        if p != 0:
            entropy -= p * torch.log(p)
    return float(entropy.cpu())


def grafting(epoch):
    while not os.path.exists('%s/ckpt%d_%d.t7' % (args.s, args.i - 1, epoch)):
        time.sleep(10)
    while True:
        try:
            checkpoint = torch.load('%s/ckpt%d_%d.t7' % (args.s, args.i - 1, epoch))
            last_net.load_state_dict(checkpoint['net'])
            break
        except:
            time.sleep(10)
    for m1, m2 in zip(net.modules(), last_net.modules()):
        if isinstance(m1, nn.Conv2d):
            w = round(args.a / np.pi * np.arctan(args.c * (entropy(m1.weight.data) - entropy(m2.weight.data))) + 0.5, 2)
            m1.weight.data = m1.weight.data.clone() * w + m2.weight.data.clone() * (1 - w)
        if isinstance(m1, nn.BatchNorm2d):
            m1.weight.data = m1.weight.data.clone() * w + m2.weight.data.clone() * (1 - w)
            m1.bias.data = m1.bias.data.clone() * w + m2.bias.data.clone() * (1 - w)
            m1.running_var.data = m1.running_var.data.clone() * w + m2.running_var.data.clone() * (1 - w)
            m1.running_mean.data = m1.running_mean.data.clone() * w + m2.running_mean.data.clone() * (1 - w)


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
    acc = 100. * correct / total
    accuracy.append(acc)
    print('Network:%d    epoch:%d    accuracy:%.3f    best:%.3f' % (args.i, epoch, acc, np.max(accuracy)))


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
        lr_scheduler.step()


if __name__ == '__main__':
    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        test(epoch)
        state = {
            'net': net.state_dict(),
            'acc': accuracy
        }
        torch.save(state, '%s/ckpt%d_%d.t7' % (args.s, args.i % args.num, epoch))
        if args.level == 'filter':
            BN(epoch)
        elif args.level == 'layer':
            grafting(epoch)
    torch.save(accuracy, '%s/accuracy%d.t7' % (args.s, args.i))
