"""
All things related to training models.
"""

import logging

import torch
import numpy as np

import octopus.handlers.devicehandler as dh


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
        inputs, targets = dh.move_data_to_device(device, model, inputs, targets)

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

    logging.info('Training iteration is finished.')
    return training_loss


def evaluate_model_on_accuracy(val_loader, model, criterion_func, device, hit_func):
    logging.info('Evaluating model...')
    training_loss = 0
    hits = 0

    with torch.no_grad():  # deactivate autograd engine to improve efficiency

        # Set model in validation mode
        model.eval()

        # process mini-batches
        for (inputs, targets) in val_loader:
            # prep
            inputs, targets = dh.move_data_to_device(device, model, inputs, targets)

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

        # calculate evaluation metrics
        training_loss /= len(val_loader)  # average per mini-batch
        hits /= len(val_loader.dataset)  # global accuracy

        logging.info('Evaluation iteration is finished.')
        return hits, training_loss


def test_model(test_loader, model, device):
    logging.info('Testing model...')
    output = []

    with torch.no_grad():  # deactivate autograd engine to improve efficiency

        # Set model in validation mode
        model.eval()

        # process mini-batches
        for batch in test_loader:

            if type(batch) is tuple:
                # loader contains inputs and targets
                inputs = batch[0]
                targets = batch[1]
            else:
                # loader contains only inputs
                inputs = batch
                targets = None

            # prep
            inputs, targets = dh.move_data_to_device(device, model, inputs, targets)

            # forward pass
            out = model.forward(inputs)

            # capture output for mini-batch
            out = out.cpu().detach().numpy()  # extract from gpu
            output.append(out)

    logging.info('Testing is finished.')
    return np.concatenate(output, axis=0)


def perform_early_stop(stats):
    logging.info('Checking early stopping criteria...')

    stopping_criteria_met = False

    if stopping_criteria_met:
        logging.info('Early stopping criteria is met.')
    else:
        logging.info('Early stopping criteria is not met.')

    return stopping_criteria_met
