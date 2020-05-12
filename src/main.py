import helper

import argparse
import torch

import data_handler
import trainer
import networks
import evaluation


def read_params_and_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prep_data', action='store_true')  # data preparation (download and extraction)
    parser.add_argument('--use_comet', action='store_true')  # if used, the experiment would be tracked by comet
    parser.add_argument('--prev_exp_id', type=str)  # previous Comet experiment id

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--epoch', type=int)

    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--checkpoints_path', type=str)
    parser.add_argument('--results_path', type=str)

    parser.add_argument('--model', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--freezed', action='store_true')

    args = parser.parse_args()
    parameters = helper.read_params('../params.json')
    return args, parameters


def adjust_params(args, params):
    if args.data_folder:
        params['data_folder'] = args.data_folder

    if args.checkpoints_path:
        params['checkpoints_path'] = args.checkpoints_path

    if args.results_path:
        params['results_path'] = args.results_path

    if args.freezed:
        params['freezed'] = args.freezed

    if args.lr:
        params['lr'] = args.lr

    print(f'In [adjust_params]: adjusted param based on args.')
    return params


def main():
    args, params = read_params_and_args()
    params = adjust_params(args, params)

    # prepare data (download and extract)
    if args.prep_data:
        data_handler.prepare_data(params)

    # evaluation
    elif args.eval:
        evaluation.evaluate_model(args, params)

    else:
        model = networks.init_unified_net(args.model, params)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
        tracker = helper.init_comet(args, params) if args.use_comet else None

        trainer.train(args, params, model, optimizer, tracker)


if __name__ == '__main__':
    main()
