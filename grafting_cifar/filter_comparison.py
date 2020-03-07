import torch
import argparse

parser = argparse.ArgumentParser(description='PyTorch Grafting Training')
parser.add_argument('--r', nargs='+', type=str)
parser.add_argument('--threshold', nargs='+', type=float)
args = parser.parse_args()


def valid_num(state_dict, threshold):
    norm = torch.cat(
        [torch.norm(values.reshape(values.shape[0], -1), p=1, dim=1) for key, values in state_dict.items() if
         'conv' in key])
    print('Total filters number:\t', len(norm))
    print('invalid filters number:\t', int(sum((norm < threshold).double())))
    print('ratio:\t', int(sum((norm < threshold).double())) / len(norm))


for threshold in args.threshold:
    print('threshold:', threshold)
    for i, dir in enumerate(args.r):
        print(dir)
        state_dict = torch.load(dir)['net']
        valid_num(state_dict, threshold)
