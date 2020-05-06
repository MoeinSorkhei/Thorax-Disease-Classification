from .unified import UnifiedNetwork


def init_unified_net(model_name, params):
    if model_name == 'resnet':
        transition_params = {'pool_mode': params['pool_mode'], 'in_channels': 512, 's': 8}
        return UnifiedNetwork(model_name, transition_params, params['freezed'])

