def CUB200_get_args_parser(subparsers):
    subparsers.add_argument('--dataset', type=str, default='cub')
    subparsers.add_argument('--device', type=str, default='cuda')


    # distributed training parameters
    subparsers.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
    subparsers.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    subparsers.add_argument('--gpu', default=[2, 3], type=list)
    subparsers.add_argument('--distributed', type=bool, default=True)

    # Tensorboard
    subparsers.add_argument('--tensorboard', default=True, type=bool)