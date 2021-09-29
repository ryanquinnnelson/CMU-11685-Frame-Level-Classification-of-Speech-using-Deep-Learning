import logging
import torch


class Training:

    def __init__(self, model, optimizer, train_loader, criterion_func, devicehandler, num_epochs):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.criterion_func = criterion_func
        self.devicehandler = devicehandler
        self.num_epochs = num_epochs

    def train_model(self, epoch):
        logging.info(f'Running epoch {epoch}/{self.num_epochs} of training...')
        train_loss = 0

        # Set model in 'Training mode'
        self.model.train()

        # process mini-batches
        for i, (inputs, targets) in enumerate(self.train_loader):
            # prep
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            inputs, targets = self.devicehandler.move_data_to_device(self.model, inputs, targets)

            # compute forward pass
            out = self.model.forward(inputs)

            # calculate loss
            loss = self.criterion_func(out, targets)
            train_loss += loss.item()

            # compute backward pass
            loss.backward()

            # update model weights
            self.optimizer.step()

            # delete mini-batch data from device
            del inputs
            del targets

        # calculate average loss across all mini-batches
        train_loss /= len(self.train_loader)

        return train_loss
