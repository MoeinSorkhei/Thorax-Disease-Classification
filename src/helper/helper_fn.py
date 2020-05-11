import os
import json
import torch
from globals import device


def read_params(params_path):
    with open(params_path, 'r') as f:  # reading params from the json file
        parameters = json.load(f)
    return parameters


def make_dir_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print(f'In [make_dir_if_not_exists]: created path "{directory}"')


def print_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'In [print_num_params]: model total params: {total_params:,} - trainable params: {trainable_params:,}')


def scientific(float_num):
    if float_num == 1e-5:
        return '1e-5'
    elif float_num == 5e-5:
        return '5e-5'
    elif float_num == 1e-4:
        return '1e-4'
    elif float_num == 1e-3:
        return '1e-3'
    raise NotImplementedError('In [scientific]: Conversion from float to scientific str needed.')


def compute_paths(args, params):
    save_path = f"{params['checkpoints_path']}/model={args.model}/freezed={params['freezed']}"
    paths = {'save_path': save_path}
    return paths


def save_model(path_to_save, epoch, model, optimizer, loss):
    name = path_to_save + f'/epoch={epoch}.pt'
    checkpoint = {'loss': loss,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, name)
    print(f'In [save_model]: save state dict done at: "{name}"')


def load_model(path_to_load, epoch, model, optimizer=None, resume_train=False):
    name = path_to_load + f'/epoch={epoch}.pt'
    checkpoint = torch.load(name, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # putting the model in the correct mode
    if resume_train:
        model.train()
        print(f'In [load_model]: model put in train mode')

    else:
        model.eval()
        print(f'In [load_model]: model put in eval mode')
        for param in model.parameters():  # freezing the layers when using only for evaluation
            param.requires_grad = False

    print(f'In [load_model]: load state dict done from: "{name}"')
    if optimizer is not None:
        return model.to(device), optimizer
    return model.to(device), None


def print_info():
    pass
