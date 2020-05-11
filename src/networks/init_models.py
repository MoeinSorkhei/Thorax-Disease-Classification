from .unified import UnifiedNetwork
import helper
from globals import device

import sys


def init_unified_net(model_name, params):
    transition_params = {'pool_mode': params['pool_mode']}

    if model_name == 'resnet':
        transition_params['in_channels'] = 512
        transition_params['s'] = 8

    elif model_name == 'vgg':
        transition_params['in_channels'] = 512
        transition_params['s'] = 8

    elif model_name == 'googlenet':
        transition_params['in_channels'] = 1024
        transition_params['s'] = 8
    
    else:
        sys.exit(f'Error: Model name "{model_name}" not recognized.')

    net = UnifiedNetwork(model_name, transition_params, params['freezed'])

    print(f'In [init_unified_net]: initialized model with {model_name} - freezed={params["freezed"]}')
    helper.print_num_params(net)
    return net.to(device)
