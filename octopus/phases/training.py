"""
Defines the training phase of model training.
"""
__author__ = 'ryanquinnnelson'

import logging
import torch


class Training:

    def __init__(self, train_loader, criterion_func, devicehandler):
        self.train_loader = train_loader
        self.criterion_func = criterion_func
        self.devicehandler = devicehandler

    def train_model(self, epoch, num_epochs, model, optimizer):
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')
        train_loss = 0

        # Set model in 'Training mode'
        model.train()

        # process mini-batches
        for i, (inputs, targets) in enumerate(self.train_loader):
            # prep
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            inputs, targets = self.devicehandler.move_data_to_device(model, inputs, targets)

            # compute forward pass
            out = model.forward(inputs)

            # calculate loss
            loss = self.criterion_func(out, targets)
            train_loss += loss.item()

            # compute backward pass
            loss.backward()

            # update model weights
            optimizer.step()

            # delete mini-batch data from device
            del inputs
            del targets

        # calculate average loss across all mini-batches
        train_loss /= len(self.train_loader)

        return train_loss
