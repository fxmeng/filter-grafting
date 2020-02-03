# implementation of filter grafting

## CIFAR dataset

### usage

grafting.py [-h] [--lr LR] [--epochs EPOCHS] [--device DEVICE]
                   [--data DATA] [--s S] [--model MODEL] [--cifar CIFAR]
                   [--print_frequence PRINT_FREQUENCE] [--a A] [--c C]
                   [--num NUM] [--i I] [--cos] [--difflr]

PyTorch Grafting Training

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --epochs EPOCHS       total epochs for training
  --device DEVICE       cuda or cpu
  --data DATA           dataset path
  --s S                 checkpoint save path
  --model MODEL         Network used
  --cifar CIFAR         cifar10 or cifar100 dataset
  --print_frequence PRINT_FREQUENCE
                        test accuracy print frequency
  --a A                 hyperparameter a for calculate weighted average
                        coefficient
  --c C                 hyper parameter c for calculate weighted average
                        coefficient
  --num NUM             Number of Networks used for grafting
  --i I                 This program is the i th Network of all Networks
  --cos                 Use cosine annealing learning rate
  --difflr              Use different initial learning rate

### Execute example

#### Two models grafting

CUDA_VISIBLE_DEVICES=0 nohup python grafting.py --s checkpoint/grafting_cifar10_resnet32 --cifar 10  --model resnet32 --num 2 --i 1 --difflr --cos >checkpoint/grafting_cifar10_resnet32/1.log &
CUDA_VISIBLE_DEVICES=1 nohup python grafting.py --s checkpoint/grafting_cifar10_resnet32 --cifar 10  --model resnet32 --num 2 --i 2  --difflr --cos >checkpoint/grafting_cifar10_resnet32/2.log &

#### Three models grafting

CUDA_VISIBLE_DEVICES=0 nohup python grafting.py --s checkpoint/grafting_cifar10_resnet32 --cifar 10  --model resnet32 --num 3 --i 1 --difflr --cos >checkpoint/grafting_cifar10_resnet32/1.log &
CUDA_VISIBLE_DEVICES=1 nohup python grafting.py --s checkpoint/grafting_cifar10_resnet32 --cifar 10  --model resnet32 --num 3 --i 2 --difflr --cos >checkpoint/grafting_cifar10_resnet32/2.log &
CUDA_VISIBLE_DEVICES=2 nohup python grafting.py --s checkpoint/grafting_cifar10_resnet32 --cifar 10  --model resnet32 --num 3 --i 3 --difflr --cos >checkpoint/grafting_cifar10_resnet32/3.log &