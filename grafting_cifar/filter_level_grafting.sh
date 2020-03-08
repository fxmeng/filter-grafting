mkdir -pv checkpoint/baseline_cifar10_vgg16;
CUDA_VISIBLE_DEVICES=0 nohup python filter_level_grafting.py --level baseline --s checkpoint/baseline_cifar10_vgg16 --cifar 10  --model vgg16 >checkpoint/baseline_cifar10_vgg16/1.log &
mkdir -pv checkpoint/baseline_cifar100_vgg16;
CUDA_VISIBLE_DEVICES=1 nohup python filter_level_grafting.py --level baseline --s checkpoint/baseline_cifar100_vgg16 --cifar 100  --model vgg16 >checkpoint/baseline_cifar100_vgg16/1.log &

mkdir -pv checkpoint/grafting_cifar10_vgg16;
CUDA_VISIBLE_DEVICES=0 nohup python filter_level_grafting.py --level layer --s checkpoint/grafting_cifar10_vgg16 --cifar 10  --model vgg16 --num 2 --i 1 >checkpoint/grafting_cifar10_vgg16/1.log &
CUDA_VISIBLE_DEVICES=1 nohup python filter_level_grafting.py --level layer --s checkpoint/grafting_cifar10_vgg16 --cifar 10  --model vgg16 --num 2 --i 2 >checkpoint/grafting_cifar10_vgg16/2.log &
mkdir -pv checkpoint/grafting_cifar100_vgg16;
CUDA_VISIBLE_DEVICES=2 nohup python filter_level_grafting.py --level layer --s checkpoint/grafting_cifar100_vgg16 --cifar 100  --model vgg16 --num 2 --i 1 >checkpoint/grafting_cifar100_vgg16/1.log &
CUDA_VISIBLE_DEVICES=3 nohup python filter_level_grafting.py --level layer --s checkpoint/grafting_cifar100_vgg16 --cifar 100  --model vgg16 --num 2 --i 2 >checkpoint/grafting_cifar100_vgg16/2.log &

mkdir -pv checkpoint/BN_grafting_cifar10_vgg16;
CUDA_VISIBLE_DEVICES=0 nohup python filter_level_grafting.py --level filter --s checkpoint/BN_grafting_cifar10_vgg16 --cifar 10  --model vgg16 --num 2 --i 1 >checkpoint/BN_grafting_cifar10_vgg16/1.log &
CUDA_VISIBLE_DEVICES=1 nohup python filter_level_grafting.py --level filter --s checkpoint/BN_grafting_cifar10_vgg16 --cifar 10  --model vgg16 --num 2 --i 2 >checkpoint/BN_grafting_cifar10_vgg16/2.log &
mkdir -pv checkpoint/BN_grafting_cifar100_vgg16;
CUDA_VISIBLE_DEVICES=2 nohup python filter_level_grafting.py --level filter --s checkpoint/BN_grafting_cifar100_vgg16 --cifar 100  --model vgg16 --num 2 --i 1 >checkpoint/BN_grafting_cifar100_vgg16/1.log &
CUDA_VISIBLE_DEVICES=3 nohup python filter_level_grafting.py --level filter --s checkpoint/BN_grafting_cifar100_vgg16 --cifar 100  --model vgg16 --num 2 --i 2 >checkpoint/BN_grafting_cifar100_vgg16/2.log &
