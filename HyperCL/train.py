import warnings

from parse_config import parse_cmd_arguments
from utils import sim_utils as sutils
from argparse import Namespace
from dataloader.data_loaders import load_datasets
from model.model import get_main_model

warnings.filterwarnings('ignore')


def main(config, shared):
    dhandlers = load_datasets(config, shared, logger,
                              data_dir='/media/dorosee/MT_RC06/yunseok/data/CL_milestone_dataset')

    # Create main network
    mnet = get_main_model(config, shared, logger, device,
                                 no_weights=not config.mnet_only)


if __name__ == "__main__":
    experiment = 'resnet'
    config = parse_cmd_arguments(mode='resnet_cifar')

    device, writer, logger = sutils.setup_environment(config,
                                                      logger_name='det_cl_cifar_%s' % experiment)

    shared = Namespace()
    shared.experiment = experiment

    main(config=config, shared=shared)

    print("Finish")
