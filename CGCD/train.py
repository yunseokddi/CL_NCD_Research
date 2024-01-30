import argparse
import sys
import torch.backends.cudnn as cudnn
import torch
import warnings

from parse_config import CUB200_get_args_parser

warnings.filterwarnings('ignore')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser('CGCD training and evaluation configs')
    config = parser.parse_known_args()[-1][0]