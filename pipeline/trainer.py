import logging
from datetime import datetime
import os

import torch
import numpy as np

import pipeline.devicedealer as devicedealer


def train_model(train_loader, model, optimizer, criterion_func, device):
    logging.info('Training model...')
    training_loss = 0

    # Set model in 'Training mode'
    model.train()

    # process mini-batches
    for i, (inputs, targets) in enumerate(train_loader):
        # prep
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        inputs, targets = devicedealer.move_data_to_device(model, device, inputs, targets)

        # compute forward pass
        out = model.forward(inputs)

        # calculate loss
        loss = criterion_func(out, targets)
        training_loss += loss.item()

        # compute backward pass
        loss.backward()

        # update model weights
        optimizer.step()

        # delete mini-batch data
        del inputs
        del targets

    # calculate average loss across all mini-batches
    training_loss /= len(train_loader)

    logging.info('Training is finished.')
    return training_loss


def evaluate_model(val_loader, model, criterion_func, device, hit_func):
    logging.info('Evaluating model...')
    training_loss = 0
    hits = 0

    with torch.no_grad():  # deactivate autograd engine to improve efficiency

        # Set model in validation mode
        model.eval()

        # process mini-batches
        for (inputs, targets) in val_loader:
            # prep
            inputs, targets = devicedealer.move_data_to_device(model, device, inputs, targets)

            # forward pass
            out = model.forward(inputs)

            # calculate validation loss
            loss = criterion_func(out, targets)
            training_loss += loss.item()

            # calculate number of accurate predictions for this batch
            out = out.cpu().detach().numpy()  # extract from gpu
            hits += hit_func(out, targets)

            # delete mini-batch
            del inputs
            del targets

        # calculate eval metrics
        training_loss /= len(val_loader)  # average per mini-batch
        hits /= len(val_loader.dataset)  # global accuracy

        logging.info('Evaluation is finished.')
        return hits, training_loss


def test_model(test_loader, model, device):
    logging.info('Testing model...')
    output = []

    with torch.no_grad():  # deactivate autograd engine to improve efficiency

        # Set model in validation mode
        model.eval()

        # process mini-batches
        for inputs in test_loader:
            # prep
            inputs, targets = devicedealer.move_data_to_device(model, device, inputs, targets=None)

            # forward pass
            out = model.forward(inputs)

            # capture output for mini-batch
            out = out.cpu().detach().numpy()  # extract from gpu
            output.append(out)

    logging.info('Testing is finished.')
    return np.concatenate(output, axis=0)


def save_results(predictions, name, results_dir):
    filename = name + '.' + datetime.now().strftime("%Y%m%d.%H.%M.%S.") + 'results.csv'

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    path = os.path.join(results_dir, filename)
    np.savetxt(path, predictions, delimiter=',', fmt='%.0f')
    logging.info(f'Saved results in:{path}.')


def perform_early_stop(stats):
    logging.info('Checking early stopping criteria...')

    perform_early_stop = False

    return perform_early_stop