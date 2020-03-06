mkdir -pv checkpoint/baseline_cifar10_resnet32;
CUDA_VISIBLE_DEVICES=0 nohup python baseline.py --s checkpoint/baseline_cifar10_resnet32 --cifar 10  --model resnet32 >checkpoint/baseline_cifar10_resnet32/1.log &
mkdir -pv checkpoint/baseline_cifar10_mobilenetv2;
CUDA_VISIBLE_DEVICES=1 nohup python baseline.py --s checkpoint/baseline_cifar10_mobilenetv2 --cifar 10  --model mobilenetv2 >checkpoint/baseline_cifar10_mobilenetv2/1.log &
