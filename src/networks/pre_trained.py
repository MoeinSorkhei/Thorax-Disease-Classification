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

    if verbose:
        print('ResNet summary before removing the last two layers')
        # print(resnet_model)
        summary(resnet_model, input_size=(3, 256, 256))

    # removing the last two layers: max pooling and linear layers
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-2])

    if verbose:
        print('ResNet summary after the last two layers')
        summary(resnet_model, input_size=(3, 256, 256))

    # freeze the resnet
    if freeze_params:
        for param in resnet_model.parameters():  # freezing the parameters
            param.requires_grad = False

    # helper.print_num_params(resnet_model)
    return resnet_model


def load_vgg(freeze_params=False, verbose=False):
    """
    Loads the VGG-16 model with batch normalization.

    :param freeze_params: whether to freeze the parameters of VGG-16 while training the unified model.
    :param verbose: if True, model summaries will be printed after loading.
    :return: the pretrained vgg-16 model.

    Notes:
        - Before removing the FC-layers, the model is massive (528 MB with 138 million parameters).
          After removing them, it is instead 56 MB with 15 million params.
          However, the forward/backward pass size is still quite big at 420 MB.

        - For 256x256 images:
            - Output shape of the forward pass is [512 x 8 x 8] excluding the batch size.

        - It would be possible to also remove the MaxPool from the end. The output shape
          would then become [512 x 16 x 16].
    """
    vgg_model = models.vgg16_bn(pretrained=True);

    if verbose:
        print('Summary of the full VGG-16 model.')
        summary(vgg_model, input_size=(3, 256, 256))

    # Removes the AvgPool and 3 fully connected layers from the end.
    vgg_model = torch.nn.Sequential(*list(vgg_model.children())[:-2])

    if verbose:
        print('\nVGG-16 summary with fully connected layers removed.')
        summary(vgg_model, input_size=(3, 256, 256))

    if freeze_params:
        for param in vgg_model.parameters():
            param.requires_grad = False

    return vgg_model


def load_googlenet(freeze_params=False, verbose=False):
    """

    :param freeze_params: if True, parameters of the GoogLeNet will be frozen and do not contribute to gradient updates.
    :param verbose: if True, the function prints model summaries before and after removing the three last layers.
    :return: the pre-trained GoogLeNet model with the three last layers (pooling and fully connected) removed.

    Notes:
        - For 256x256 images:
            - Output shape of the forward pass is [1024 x 8 x 8] excluding the batch size for GoogLeNet.
    """
    
    googlenet_model = models.googlenet(pretrained=True)
    googlenet_model.train()  # put the model in the correct mode

    if verbose:
        print('GoogLeNet summary before removing the last three layers')
        # print(googlenet_model)
        summary(googlenet_model, input_size=(3, 256, 256))

    # removing the last three layers: AdaptiveAvgPool2d, Dropout and Linear
    googlenet_model = torch.nn.Sequential(*list(googlenet_model.children())[:-3])

    if verbose:
        print('GoogLeNet summary after the last three layers')
        summary(googlenet_model, input_size=(3, 256, 256))

    # freeze the GoogLeNet
    if freeze_params:
        for param in googlenet_model.parameters():  # freezing the parameters
            param.requires_grad = False

    # helper.print_num_params(googlenet_model)
    return googlenet_model


def load_pre_trained_model(model_name, freezed):
    if model_name == 'resnet':
        return load_resnet(freeze_params=freezed)
    
    elif model_name == 'vgg':
        return load_vgg(freeze_params=freezed)
    
    elif model_name == 'googlenet':
        return load_googlenet(freeze_params=freezed)