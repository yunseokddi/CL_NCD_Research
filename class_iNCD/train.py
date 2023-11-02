import argparse
import sys
import torch.backends.cudnn as cudnn
import warnings
import torch
import numpy as np
import random
import os
import torch.nn as nn
import copy
import time
import datetime

from parse_config import CIFAR100_get_args_parser
from tensorboard_logger import configure
from utils.utils import init_distributed_mode, BCE
from dataloader.cifarloader import CIFAR100Loader, CIFAR100LoaderMix
from model.resnet import ResNet, BasicBlock
from trainer.trainer import FirstTrainer
from timm.optim import create_optimizer
from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def main(args):
    init_distributed_mode(args)

    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # print(args)

    if args.mode == 'train' and args.step == 'first':
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes1

        mix_train_loader = CIFAR100LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                             aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes),
                                             unlabeled_list=range(args.num_labeled_classes, num_classes))
        unlabeled_val_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug=None,
                                              shuffle=False, target_list=range(args.num_labeled_classes, num_classes))
        unlabeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                               aug=None,
                                               shuffle=False, target_list=range(args.num_labeled_classes, num_classes))
        labeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug=None, shuffle=True, target_list=range(args.num_labeled_classes))
        labeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                             shuffle=False, target_list=range(args.num_labeled_classes))
        all_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                         shuffle=False, target_list=range(num_classes))

        model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes,
                       args.num_unlabeled_classes1 + args.num_unlabeled_classes2).to(device)
        model.to(device)

        state_dict = torch.load(args.warmup_model_dir)
        model.load_state_dict(state_dict, strict=False)

        # Unlabelled data's classifier
        model.head2 = nn.Linear(512, args.num_unlabeled_classes1).to(device)

        for name, param in model.named_parameters():
            if 'head' not in name and 'layer4' not in name and 'layer3' not in name and 'layer2' not in name:
                param.requires_grad = False

        if args.w_kd > 0:
            old_model = copy.deepcopy(model)
            old_model = old_model.to(device)

            if args.distributed:
                old_model = torch.nn.parallel.DistributedDataParallel(old_model, device_ids=[args.gpu])
            old_model.eval()
        else:
            old_model = None

        save_weight = model.head1.weight.data.clone()  # save the weights of head-1
        save_bias = model.head1.bias.data.clone()  # save the bias of head-1
        model.head1 = nn.Linear(512, num_classes).to(device)  # replace the labeled-class only head-1
        model.head1.weight.data[:args.num_labeled_classes] = save_weight  # put the old weights into the old part
        model.head1.bias.data[:] = torch.min(save_bias) - 1.  # put the bias
        model.head1.bias.data[:args.num_labeled_classes] = save_bias

        model_without_ddp = model

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(args, model)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        criterion_ce = nn.CrossEntropyLoss()  # CE loss for labeled data
        criterion_bce = BCE()  # BCE loss for unlabeled data

        first_trainer = FirstTrainer(args, model, model_without_ddp, old_model, labeled_train_loader, mix_train_loader,
                                     labeled_test_loader,
                                     unlabeled_val_loader,
                                     all_test_loader,
                                     optimizer,
                                     scheduler, criterion_ce, criterion_bce)

        start = time.time()

        first_trainer.train()

        result_sec = time.time()- start
        print("Total training time : {}".format(datetime.timedelta(seconds=result_sec)))
        print("finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('iNCD training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100':
        config_parser = subparser.add_parser('cifar100', help='iNCD-CIFAR100 configs')
        CIFAR100_get_args_parser(config_parser)

    else:
        assert "Check dataset"

    args = parser.parse_args()

    if args.tensorboard:
        configure("runs/%s" % (args.dataset_name))

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir = os.path.join(args.exp_root, runner_name + '_' + args.dataset_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir + '/' + args.step + '_' + '{}.pth'.format(args.model_name)

    main(args=args)

    sys.exit(0)
