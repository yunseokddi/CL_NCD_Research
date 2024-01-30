from .split_cifar import get_split_cifar_handlers

def load_datasets(config, shared, logger, data_dir='../datasets'):
    augment_data = not config.disable_data_augmentation

    if augment_data:
        logger.info('Data augmentation will be used.')

    assert (config.num_tasks <= 11)
    logger.info('Loading CIFAR datasets ...')

    dhandlers = get_split_cifar_handlers(data_dir, use_one_hot=True,
                                         use_data_augmentation=augment_data, num_tasks=config.num_tasks)
    assert (len(dhandlers) == config.num_tasks)

    logger.info('Loaded %d CIFAR task(s) into memory.' % config.num_tasks)

    return dhandlers