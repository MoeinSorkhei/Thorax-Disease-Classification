import helper

import argparse
import torch

import data_handler
import trainer
import networks


def read_params_and_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prep_data', action='store_true')  # data preparation (download and extraction)
    parser.add_argument('--use_comet', action='store_true')  # if used, the experiment would be tracked by comet

    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--checkpoints_path', type=str)
    parser.add_argument('--prev_exp_id', type=str)  # previous Comet experiment id

    parser.add_argument('--model', type=str)
    parser.add_argument('--lr', type=float)

    args = parser.parse_args()
    parameters = helper.read_params('../params.json')
    return args, parameters


def adjust_params(args, params):
    if args.data_folder:
        params['data_folder'] = args.data_folder

    if args.checkpoints_path:
        params['checkpoints_path'] = args.checkpoints_path

    if args.lr:
        params['lr'] = args.lr

    print(f'In [adjust_params]: adjusted param based on args.')
    return params


def main():
    args, params = read_params_and_args()
    params = adjust_params(args, params)

    if args.prep_data:  # prepare data (download and extract)
        data_handler.prepare_data(params)

    else:
        model = networks.init_unified_net(args.model, params)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
        tracker = helper.init_comet(args, params) if args.use_comet else None

        trainer.train(args, params, model, optimizer, tracker)


if __name__ == '__main__':
    main()
