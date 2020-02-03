CUDA_VISIBLE_DEVICES=0 nohup python grafting.py --arch resnet18 --s grafting_imagenet_resnet18 --num 2 --i 1 >checkpoint/grafting_imagenet_resnet18/1.out &
CUDA_VISIBLE_DEVICES=1 nohup python grafting.py --arch resnet18 --s grafting_imagenet_resnet18 --num 2 --i 2 >checkpoint/grafting_imagenet_resnet18/2.out &
