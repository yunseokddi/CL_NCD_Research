import torch

from utils import misc
from .resnet import ResNet
from .mnet_interface import MainNetInterface
from warnings import warn

def get_main_model(config, shared, logger, device, no_weights=False):
    net_type = 'resnet'
    logger.info('Building a ResNet ...')

    num_outputs = 10

    # cl_scenario 1 : TIL
    # cl_scenario 3 : CIL
    if config.cl_scenario == 1 or config.cl_scenario == 3:
        num_outputs *= config.num_tasks

    logger.info('The network will have %d output neurons.' % num_outputs)

    in_shape = [32, 32, 3]
    out_shape = [num_outputs]

    mnet = get_mnet_model(config, net_type, in_shape, out_shape, device,
                                 no_weights=no_weights)

    init_network_weights(mnet.weights, config, logger, net=mnet)

    return mnet

def get_mnet_model(config, net_type, in_shape, out_shape, device, cprefix=None,
                   no_weights=False):
    assert (net_type in ['mlp', 'resnet', 'zenke', 'bio_conv_net'])

    if cprefix is None:
        cprefix = ''

    def gc(name):
        """Get config value with that name."""
        return getattr(config, '%s%s' % (cprefix, name))
    def hc(name):
        """Check whether config exists."""
        return hasattr(config, '%s%s' % (cprefix, name))

    mnet = None

    if hc('net_act'):
        net_act = gc('net_act')
        net_act = misc.str_to_act(net_act)
    else:
        net_act = None

    def get_val(name):
        ret = None
        if hc(name):
            ret = gc(name)
        return ret

    no_bias = get_val('no_bias')
    dropout_rate = get_val('dropout_rate')
    specnorm = get_val('specnorm')
    batchnorm = get_val('batchnorm')
    no_batchnorm = get_val('no_batchnorm')
    bn_no_running_stats = get_val('bn_no_running_stats')
    bn_distill_stats = get_val('bn_distill_stats')

    use_bn = None
    if batchnorm is not None:
        use_bn = batchnorm
    elif no_batchnorm is not None:
        use_bn = not no_batchnorm

    assign = lambda x, y: y if x is None else x

    assert (len(out_shape) == 1)

    mnet = ResNet(in_shape=in_shape, num_classes=out_shape[0],
                  verbose=True,  # n=5,
                  no_weights=no_weights,
                  # init_weights=None,
                  use_batch_norm=assign(use_bn, True),
                  bn_track_stats=assign(not bn_no_running_stats, True),
                  distill_bn_stats=assign(bn_distill_stats, False),
                  # use_context_mod=False,
                  # context_mod_inputs=False,
                  # no_last_layer_context_mod=False,
                  # context_mod_no_weights=False,
                  # context_mod_post_activation=False,
                  # context_mod_gain_offset=False,
                  # context_mod_apply_pixel_wise=False
                  ).to(device)

    return mnet

def get_hnet_model(config, mnet, logger, device):
    logger.info('Creating hypernetwork ...')


def build_hnet_model(config, num_tasks, device, mnet_shapes, cprefix=None):
    if cprefix is None:
        cprefix = ''

    def gc(name):
        """Get config value with that name."""
        return getattr(config, '%s%s' % (cprefix, name))

    hyper_chunks = misc.str_to_ints(gc('hyper_chunks'))
    assert(len(hyper_chunks) in [1,2,3])
    if len(hyper_chunks) == 1:
        hyper_chunks = hyper_chunks[0]

    hnet_arch = misc.str_to_ints(gc('hnet_arch'))
    sa_hnet_filters = misc.str_to_ints(gc('sa_hnet_filters'))
    sa_hnet_kernels = misc.str_to_ints(gc('sa_hnet_kernels'))
    sa_hnet_attention_layers = misc.str_to_ints(gc('sa_hnet_attention_layers'))

    hnet_act = misc.str_to_act(gc('hnet_act'))

    if isinstance(hyper_chunks, list): # Chunked self-attention hypernet
        if len(sa_hnet_kernels) == 1:
            sa_hnet_kernels = sa_hnet_kernels[0]
        # Note, that the user can specify the kernel size for each dimension and
        # layer separately.
        elif len(sa_hnet_kernels) > 2 and \
            len(sa_hnet_kernels) == gc('sa_hnet_num_layers') * 2:
            tmp = sa_hnet_kernels
            sa_hnet_kernels = []
            for i in range(0, len(tmp), 2):
                sa_hnet_kernels.append([tmp[i], tmp[i+1]])

        if gc('hnet_dropout_rate') != -1:
            warn('SA-Hypernet doesn\'t use dropout. Dropout rate will be ' +
                 'ignored.')
        if gc('hnet_act') != 'relu':
            warn('SA-Hypernet doesn\'t support the other non-linearities ' +
                 'than ReLUs yet. Option "%shnet_act" (%s) will be ignored.'
                 % (cprefix, gc('hnet_act')))


def init_network_weights(all_params, config, logger, chunk_embs=None,
                         task_embs=None, net=None):
    """Initialize a given set of weight tensors according to the user
    configuration.

    Warning:
        This method is agnostic to where the weights stem from and is
        therefore slightly dangerous. Use with care.

    Note:
        The method only exists as at the time of implementation the package
        :mod:`hnets` wasn't available yet. In the future, initialization should
        be part of the network implementation (e.g., via method
        :meth:`mnets.mnet_interface.MainNetInterface.custom_init`).

    Note:
        If the given network implements interface
        :class:`mnets.mnet_interface.MainNetInterface`, then the corresponding
        method :meth:`mnets.mnet_interface.MainNetInterface.custom_init` is
        used.

    Note:
        Papers like the following show that hypernets should get a special
        init. This function does not take this into consideration.

            https://openreview.net/forum?id=H1lma24tPB

    Args:
        all_params: A list of weight tensors to be initialized.
        config: Command-line arguments.
        logger: Logger.
        chunk_embs (optional): A list of chunk embeddings.
        task_embs (optional): A list of task embeddings.
        net (optional): The network from which the parameters stem come from.
            Can be used to implement network specific initializations (e.g.,
            batch-norm weights).
    """
    if config.custom_network_init:
        if net is not None and isinstance(net, MainNetInterface):
            logger.info('Applying custom initialization to network ...')
            net.custom_init(normal_init=config.normal_init,
                            normal_std=config.std_normal_init, zero_bias=True)

        else:
            logger.warning('Custom weight initialization is applied to all ' +
                           'network parameters. Note, the current ' +
                           'implementation might be agnostic to special ' +
                           'network parameters.')
            for W in all_params:
                # FIXME not all 1D vectors are bias vectors.
                # Examples of parameters that are 1D and not bias vectors:
                # * batchnorm weights
                # * embedding vectors
                if W.ndimension() == 1: # Bias vector.
                    torch.nn.init.constant_(W, 0)
                elif config.normal_init:
                    torch.nn.init.normal_(W, mean=0, std=config.std_normal_init)
                else:
                    torch.nn.init.xavier_uniform_(W)


    # Note, the embedding vectors inside "all_params" have been considered
    # as bias vectors and thus initialized to zero.
    if chunk_embs is not None:
        for emb in chunk_embs:
            torch.nn.init.normal_(emb, mean=0, std=config.std_normal_emb)

    if task_embs is not None:
        for temb in task_embs:
            torch.nn.init.normal_(temb, mean=0., std=config.std_normal_temb)