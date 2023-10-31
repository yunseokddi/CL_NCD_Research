def CIFAR100_get_args_parser(subparsers):
    subparsers.add_argument('--gamma', type=float, default=0.1)
    subparsers.add_argument('--w_kd', type=float, default=10.0)
    subparsers.add_argument('--epochs', default=200, type=int)
    subparsers.add_argument('--rampup_length', default=150, type=int)
    subparsers.add_argument('--rampup_coefficient', type=float, default=50)
    subparsers.add_argument('--increment_coefficient', type=float, default=0.01)
    subparsers.add_argument('--step_size', default=170, type=int)
    subparsers.add_argument('--batch_size', default=128, type=int)
    subparsers.add_argument('--num_unlabeled_classes1', default=10, type=int)
    subparsers.add_argument('--num_unlabeled_classes2', default=10, type=int)
    subparsers.add_argument('--num_labeled_classes', default=80, type=int)
    subparsers.add_argument('--dataset_root', type=str, default='/media/dorosee/MT_RC06/yunseok/data/class_iNCD/datasets/CIFAR/')
    subparsers.add_argument('--exp_root', type=str, default='./checkpoint/')
    subparsers.add_argument('--warmup_model_dir', type=str,
                        default='./checkpoint/supervised_learning_wo_ssl/warmup_cifar100_resnet_wo_ssl.pth')
    subparsers.add_argument('--finetune_model_dir', type=str,
                        default='./data/experiments/pretrain/auto_novel/resnet_rotnet_cifar10.pth')
    subparsers.add_argument('--topk', default=5, type=int)
    # parser.add_argument('--IL', action='store_true', default=False, help='w/ incremental learning')
    subparsers.add_argument('--IL_version', type=str, default='OG', choices=['OG', 'LwF', 'LwFProto', 'JointHead1',
                                                                         'JointHead1woPseudo', 'SplitHead12',
                                                                         'OGwoKD', 'OGwoProto', 'OGwoPseudo',
                                                                         'AutoNovel', 'OGwoKDwoProto', 'OGwoKDwoPseudo',
                                                                         'OGwoProtowoPseudo', 'OGwoKDwoProtowoPseudo'])
    subparsers.add_argument('--detach_B', action='store_true', default=False, help='Detach the feature of the backbone')
    subparsers.add_argument('--l2_classifier', action='store_true', default=False, help='L2 normalize classifier')
    subparsers.add_argument('--labeled_center', type=float, default=10.0)
    subparsers.add_argument('--model_name', type=str, default='FRoST_1st_OG_kd10_p1_cifar100')
    subparsers.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, svhn')
    subparsers.add_argument('--seed', default=10, type=int)
    subparsers.add_argument('--mode', type=str, default='train')
    subparsers.add_argument('--lambda_proto', type=float, default=1.0, help='weight for the source prototypes loss')
    subparsers.add_argument('--step', type=str, default='first', choices=['first', 'second'])
    subparsers.add_argument('--first_step_dir', type=str,
                        default='./data/experiments/incd_2step_cifar100_cifar100/first_FRoST_1st_OG_kd10_p1_cifar100.pth')

    subparsers.add_argument('--device', type=str, default='cuda')

    # first stage optimizer parameters
    subparsers.add_argument('--opt', type=str, default='sgd')
    subparsers.add_argument('--momentum', type=float, default=0.9)
    subparsers.add_argument('--weight_decay', type=float, default=1e-4)
    subparsers.add_argument('--lr', type=float, default=0.1)


    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    subparsers.add_argument('--gpu', default=[2, 3], type=list)
    subparsers.add_argument('--distributed', type=bool, default=True)

    # Tensorboard
    subparsers.add_argument('--tensorboard', default=True, type=bool)