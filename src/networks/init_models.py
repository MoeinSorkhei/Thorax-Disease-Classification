from .unified import UnifiedNetwork

import sys

def init_unified_net(model_name, params):
    transition_params = {'pool_mode': params['pool_mode']}

    if model_name == 'resnet':
        transition_params['in_channels'] = 512
        transition_params['s'] = 8

    elif model_name == 'vgg':
        transition_params['in_channels'] = 512
        transition_params['s'] = 8
    
    else:
        sys.exit(f'Error: Model name "{model_name}" not recognized.')

    return UnifiedNetwork(model_name, transition_params, params['freezed'])