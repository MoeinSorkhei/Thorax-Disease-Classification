import numpy as np
import torch

import data_handler
import helper
import networks
from globals import device, pathology_names
from . import plotting


def evaluate_model(args, params):
    loader_params = {
        'batch_size': params['batch_size'],
        'shuffle': False,
        'num_workers': params['num_workers']
    }
    batch_size = params['batch_size']

    _, _, test_loader = data_handler.init_data_loaders(params, loader_params)

    total_predicted = np.zeros((batch_size, 14))
    total_labels = np.zeros((batch_size, 14))

    path_to_load = helper.compute_paths(args, params)['save_path']  # compute path from args and params
    net = networks.init_unified_net(args.model, params)
    net, _ = helper.load_model(path_to_load, args.epoch, net, optimizer=None, resume_train=False)

    print(f'In [evaluate_model]: loading the model done from: "{path_to_load}"')
    print(f'In [evaluate_model]: starting evaluation with {len(test_loader)} batches')

    with torch.no_grad():
        for i_batch, batch in enumerate(test_loader):
            img_batch = batch['image'].to(device).float()
            label_batch = batch['label'].to(device).float()
            pred = net(img_batch)

            if i_batch > 0:
                total_predicted = np.append(total_predicted, pred.cpu().detach().numpy(), axis=0)
                total_labels = np.append(total_labels, label_batch.cpu().detach().numpy(), axis=0)
            else:
                total_predicted = pred.cpu().detach().numpy()
                total_labels = label_batch.cpu().detach().numpy()

            if i_batch % 50 == 0:
                print(f'In [evaluate_model]: prediction done for batch {i_batch}')

    results_path = helper.compute_paths(args, params)['results_path']
    helper.make_dir_if_not_exists(results_path)

    # plot roc
    print(f'In [evaluate_model]: starting plotting ROC...')
    plotting.plot_roc(total_predicted, total_labels, pathology_names, results_path)
