import argparse
import torch

import data_handler
import helper
import trainer
import networks


def read_params_and_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prep_data', action='store_true')  # data preparation (download and extraction)
    parser.add_argument('--use_comet', action='store_true')  # if used, the experiment would be tracked by comet

    parser.add_argument('--model', type=str)

    args = parser.parse_args()
    parameters = helper.read_params('../params.json')
    return args, parameters


def adjust_params(args, params):
    pass


def main():
    args, params = read_params_and_args()

    if args.prep_data:  # prepare data (download and extract)
        data_handler.prepare_data(params)

    else:
        model = networks.init_unified_net(args.model, params)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
        trainer.train(args, params, model, optimizer)


if __name__ == '__main__':
    main()
