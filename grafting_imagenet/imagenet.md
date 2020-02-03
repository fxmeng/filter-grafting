# implementation of filter grafting

## ImageNet dataset

### usage

grafting.py [-h] [--data DIR] [-a ARCH] [-j N] [--epochs N]
                   [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--wd W]
                   [-p N] [--resume RESUME] [-e] [--pretrained] [--gpu GPU]
                   [--s S] [--num NUM] [--i I] [--a A] [--c C]

PyTorch ImageNet Training

optional arguments:
  -h, --help            show this help message and exit
  --data DIR            path to dataset
  -a ARCH, --arch ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | inception_v3
                        | resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume RESUME       path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --gpu GPU             GPU id to use.
  --s S                 checkpoint save dir
  --num NUM             number of Networks in grafting
  --i I                 the i-th program
  --a A                 hyperparameter a for calculate weighted average
                        coefficient
  --c C                 hyper parameter c for calculate weighted average
                        coefficient

### Execute example

#### Two models grafting

CUDA_VISIBLE_DEVICES=0 nohup python grafting.py --arch resnet18 --s grafting_imagenet_resnet18 --num 2 --i 1 >checkpoint/grafting_imagenet_resnet18/1.out &
CUDA_VISIBLE_DEVICES=1 nohup python grafting.py --arch resnet18 --s grafting_imagenet_resnet18 --num 2 --i 2 >checkpoint/grafting_imagenet_resnet18/2.out &

#### Three models grafting

CUDA_VISIBLE_DEVICES=0 nohup python grafting.py --arch resnet18 --s grafting_imagenet_resnet18 --num 3 --i 1 >checkpoint/grafting_imagenet_resnet18/1.out &
CUDA_VISIBLE_DEVICES=1 nohup python grafting.py --arch resnet18 --s grafting_imagenet_resnet18 --num 3 --i 2 >checkpoint/grafting_imagenet_resnet18/2.out &CUDA_VISIBLE_DEVICES=2 nohup python grafting.py --arch resnet18 --s grafting_imagenet_resnet18 --num 3 --i 3 >checkpoint/grafting_imagenet_resnet18/3.out &