import torch

import data_handler, helper
from . import loss

# this device is accessible in all the functions in this file
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(args, params, model, optimizer, tracker=None):
    loader_params = {
        'batch_size': params['batch_size'],
        'shuffle': params['shuffle'],
        'num_workers': params['num_workers']
    }

    train_loader, val_loader, _ = data_handler.init_data_loaders(params, loader_params)

    epoch = 0
    max_epochs = params['max_epochs']

    while epoch < max_epochs:
        print(f'{"=" * 40} In epoch: {epoch} {"=" * 40}')
        print(f'Training on {len(train_loader)} batches...')

        # each for loop for one epoch
        for i_batch, batch in enumerate(train_loader):
            # converting the labels batch  to from Long tensor to Float tensor (otherwise won't work on GPU)
            img_batch = batch['image'].to(device).float()
            label_batch = batch['label'].to(device).float()

            # making gradients zero in each optimization step
            optimizer.zero_grad()

            # getting the network prediction and computing the loss
            pred = model(img_batch)
            train_loss = loss.compute_wcel(fx=pred, labels=label_batch)
            # if i_batch % 50 == 0:
            print(f'Batch: {i_batch}, train loss: {round(train_loss.item(), 3)}')

            # tracking the metrics using comet in each iteration
            if args.use_comet:
                tracker.track_metric('train_loss', round(train_loss.item(), 3))

            # backward and optimization step
            train_loss.backward()
            optimizer.step()

        helper.save_model()  # save the model every epoch
        val_loss = loss.compute_val_loss(model, val_loader)  # track val loss
        if args.use_comet:
            tracker.track_metric('val_loss', val_loss)
        epoch += 1


