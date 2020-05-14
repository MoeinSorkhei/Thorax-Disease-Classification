import torch.nn

from . import *


class UnifiedNetwork(torch.nn.Module):
    def __init__(self, model_name, transition_params, freezed=False):
        """
        Loads the wanted pre-trained model, creates a transition layer, and then connects the loaded model to the
        TransitionLayer which contains the pooling and prediction layers in itself.

        :param model_name:
        :param transition_params:
        :param freezed:
        """
        super().__init__()
        # self.freeze_pre_trained = freeze_pre_trained
        self.pre_trained_model = load_pre_trained_model(model_name, freezed)
        self.trans_pool_pred = TransPoolPred(**transition_params)

    def forward(self, inp, return_cam=False):
        """
        :param inp:
        :param return_cam: if True, only returns class activation maps (used for heat-map generation).
        :return:
        """
        model_out = self.pre_trained_model(inp)
        prediction = self.trans_pool_pred(model_out, return_cam=return_cam)
        return prediction
