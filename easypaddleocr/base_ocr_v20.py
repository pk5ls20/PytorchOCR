import os
import torch
from .pytorchocr.modeling.architectures.base_model import BaseModel


class BaseOCRV20:
    def __init__(self, config, **kwargs):
        self.config = config
        self.net = BaseModel(self.config, **kwargs)
        self.net.eval()

    @staticmethod
    def read_pytorch_weights(weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError('{} is not existed.'.format(weights_path))
        weights = torch.load(weights_path)
        return weights

    @staticmethod
    def get_out_channels(weights):
        if list(weights.keys())[-1].endswith('.weight') and len(list(weights.values())[-1].shape) == 2:
            out_channels = list(weights.values())[-1].numpy().shape[1]
        else:
            out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        # print('weights is loaded.')

    def load_pytorch_weights(self, weights_path):
        self.net.load_state_dict(torch.load(weights_path))
        # print('model is loaded: {}'.format(weights_path))
