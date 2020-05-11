import torch

# this device is accessible in all the modules who import this module
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
