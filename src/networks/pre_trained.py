from torchvision import models
from torchsummary import summary
import torch

import helper


def load_resnet(freeze_params=False, verbose=False):
    """
    Example from https://pytorch.org/docs/stable/torchvision/models.html
    and https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html.

    :param freeze_params: if True, parameters of the resnet will be frozen and do not contribute to gradient updates.
    :param verbose: if True, the function prints model summaries before and after removing the two last layers.
    :return: the pre-trained resnet34 model with the two last layers (pooling and fully connected) removed.

    Notes:
        - For 256x256 images:
            - Output shape of the forward pass is [512 x 8 x 8] excluding the batch size for resnet34.
            - Output shape of the forward pass is [2048 x 8 x 8] excluding the batch size for resnet50.
    """
    # resnet_model = models.resnet50(pretrained=True)
    resnet_model = models.resnet34(pretrained=True)
    resnet_model.train()  # put the model in the correct mode

    # freeze the resnet
    if freeze_params:
        for param in resnet_model.parameters():  # freezing the parameters
            param.requires_grad = False

    if verbose:
        print('ResNet summary before removing the last two layers')
        # print(resnet_model)
        summary(resnet_model, input_size=(3, 256, 256))

    # removing the last two layers: max pooling and linear layers
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-2])

    if verbose:
        print('ResNet summary after the last two layers')
        summary(resnet_model, input_size=(3, 256, 256))

    # helper.print_num_params(resnet_model)
    return resnet_model


def load_pre_trained_model(model_name, freezed):
    if model_name == 'resnet':
        return load_resnet(freeze_params=freezed)
