import torch
import torch.nn as nn
import torch.nn.functional as F


class TransPoolPred(nn.Module):
    def __init__(self, pool_mode, in_channels, s):
        """
        Transition, Pooling, and Prediction layers are in this module.
        :param pool_mode: the pooling mode for the pooling layer. Currently supports 'max', 'avg', or 'max_avg'.
        :param in_channels: number of channels of the feature maps (the output of the pre-trained model).
        :param s: the spatial dimension of the feature maps.
        """
        super().__init__()
        self.pool_mode = pool_mode
        self.in_channels = in_channels
        self.s = s  # spatial size of the feature maps

        # transition layer: 1x1 Conv2D with equal and input and output channels
        self.transition = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        # prediction layer
        self.prediction = nn.Linear(in_features=in_channels, out_features=14)
        # final sigmoid layer - similar to softmax but instead sigmoid because the diseases are not exclusive
        self.sigmoid = nn.Sigmoid()

    def global_pool(self, inp):
        height, width = inp.shape[2], inp.shape[3]  # inp of shape (B, C, H, W)
        if self.pool_mode == 'max':  # squeeze after pooling: (B, C, 1, 1) -> (B, C)
            return F.max_pool2d(inp, kernel_size=(height, width)).squeeze(dim=3).squeeze(dim=2)

        elif self.pool_mode == 'avg':
            return F.avg_pool2d(inp, kernel_size=(height, width)).squeeze(dim=3).squeeze(dim=2)

        elif self.pool_mode == 'max_avg':
            max_val = F.max_pool2d(inp, kernel_size=(height, width)).squeeze(dim=3).squeeze(dim=2)
            avg_val = F.avg_pool2d(inp, kernel_size=(height, width)).squeeze(dim=3).squeeze(dim=2)
            return (max_val + avg_val) / 2

    def forward(self, inp):
        """
        The forward pass of the transition layer, for either classification or generation of heat-maps.
        :param inp:
        :return:
        """
        trans_out = self.transition(inp)
        pool_out = self.global_pool(trans_out)
        pred_out = self.prediction(pool_out)
        sigmoid_out = self.sigmoid(pred_out)
        return sigmoid_out
