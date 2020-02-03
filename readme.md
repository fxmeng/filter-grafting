# Implementation of filter grafting
I'm playing with PyTorch on the CIFAR and ImageNet dataset
## Prerequisites
Python 3.6+
PyTorch 1.0+

## CIFAR dataset

### usage

cd grafting_cifar

./grafting.sh

for more running detail see ./grafting_cifar/cifar.md

### Results

model | method | CIFAR-10 | CIFAR-100
---- | ---- | ---- | ----
ResNet32 | baseline | 92.83 | 69.8 
| grafting | 93.94 | 71.28 
ResNet56 | baseline| 93.50| 71.55
| grafting | 94.73 | 72.83 
ResNet110| baseline| 93.81| 73.21
| grafting | 94.96 | 75.27 
MobileNetV2| baseline| 92.42| 71.44
| grafting | 94.20 | 74.15 
WRN8-10| baseline | 95.75 | 80.65 
| grafting | 96.40 | 81.62 

## ImageNet dataset

### usage

cd grafting_imagenet

./grafting.sh

for more running detail see ./grafting_imagenet/imagenet.md

### result

| model    | method   | top-1 | top-5 |
| -------- | -------- | ----- | ----- |
| ResNet18 | baseline | 69.15 | 88.87 |
|          | grafting | 71.19 | 90.01 |
| ResNet34 | baseline | 72.60 | 90.91 |
|          | grafting | 74.58 | 92.05 |

