CUDA_VISIBLE_DEVICES=0 nohup python3 grafting.py --s 1 --cifar 100 --model mobilenetv2 --i 1 --lr 0.1 --threshold 0.1 --sh noise >checkpoint/1/1.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 grafting.py --s 2 --cifar 100 --model mobilenetv2 --i 2 --lr 0.1 --threshold 0.1 --sh inside >checkpoint/2/1.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 grafting.py --s 3 --cifar 100  --model mobilenetv2 --num 2 --i 1 --cos --sh entropy >checkpoint/3/1.out &
CUDA_VISIBLE_DEVICES=0 nohup python3 grafting.py --s 3 --cifar 100  --model mobilenetv2 --num 2 --i 2 --cos --sh entropy >checkpoint/3/2.out &

CUDA_VISIBLE_DEVICES=0 nohup python3 grafting.py --s 4 --cifar 100  --model mobilenetv2 --num 4 --i 1 --cos --difflr --sh entropy >checkpoint/4/1.out &
CUDA_VISIBLE_DEVICES=0 nohup python3 grafting.py --s 4 --cifar 100  --model mobilenetv2 --num 4 --i 2 --cos --difflr --sh entropy >checkpoint/4/2.out &
CUDA_VISIBLE_DEVICES=1 nohup python3 grafting.py --s 4 --cifar 100  --model mobilenetv2 --num 4 --i 3 --cos --difflr --sh entropy >checkpoint/4/3.out &
CUDA_VISIBLE_DEVICES=1 nohup python3 grafting.py --s 4 --cifar 100  --model mobilenetv2 --num 4 --i 4 --cos --difflr --sh entropy >checkpoint/4/4.out &