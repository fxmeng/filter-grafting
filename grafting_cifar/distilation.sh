CUDA_VISIBLE_DEVICES=3 nohup python3 distillation.py --s 2 --cifar 10 --model resnet20 --teacher checkpoint/5.75/ckpt7_149.t7 >checkpoint/2/7.out &
CUDA_VISIBLE_DEVICES=3 nohup python3 distillation.py --s 2 --cifar 100 --model resnet20 --teacher checkpoint/5.75/ckpt8_149.t7 >checkpoint/2/8.out &
CUDA_VISIBLE_DEVICES=3 nohup python3 distillation.py --s 2 --cifar 10 --model resnet32 --teacher checkpoint/5.75/ckpt1_149.t7 >checkpoint/2/1.out &
CUDA_VISIBLE_DEVICES=3 nohup python3 distillation.py --s 2 --cifar 100 --model resnet32 --teacher checkpoint/5.75/ckpt2_149.t7 >checkpoint/2/2.out &
CUDA_VISIBLE_DEVICES=2 nohup python3 distillation.py --s 2 --cifar 10 --model resnet56 --teacher checkpoint/5.75/ckpt3_149.t7 >checkpoint/2/3.out &
CUDA_VISIBLE_DEVICES=2 nohup python3 distillation.py --s 2 --cifar 100 --model resnet56 --teacher checkpoint/5.75/ckpt4_149.t7 >checkpoint/2/4.out &
CUDA_VISIBLE_DEVICES=1 nohup python3 distillation.py --s 2 --cifar 10 --model resnet110 --teacher checkpoint/5.75/ckpt5_149.t7 >checkpoint/2/5.out &
CUDA_VISIBLE_DEVICES=1 nohup python3 distillation.py --s 2 --cifar 100 --model resnet110 --teacher checkpoint/5.75/ckpt6_149.t7 >checkpoint/2/6.out &
CUDA_VISIBLE_DEVICES=5 nohup python3 distillation.py --s 2 --cifar 10 --model mobilenetv2 --teacher checkpoint/5.75/ckpt9_149.t7 >checkpoint/2/9.out &
CUDA_VISIBLE_DEVICES=0 nohup python3 distillation.py --s 2 --cifar 100 --model mobilenetv2 --teacher checkpoint/5.75/ckpt10_149.t7 >checkpoint/2/10.out &

