mkdir -p checkpoint/baseline_resnet50;
CUDA_VISIBLE_DEVICES=0 nohup python baseline.py --arch resnet50 --s baseline_resnet50 >checkpoint/baseline_resnet50/1.out &

