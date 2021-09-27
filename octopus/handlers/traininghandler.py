"""
All things related to training models.
"""

import logging
import time

import torch
import numpy as np

import octopus.handlers.devicehandler as dh


class TrainingHandler:

    def __init__(self, load_from_checkpoint, first_epoch, num_epochs, checkpoint_file=None):
        self.load_from_checkpoint = load_from_checkpoint
        self.checkpoint_file = checkpoint_file
        self.num_epochs = num_epochs
        self.first_epoch = first_epoch

        # setup tracking variables and early stopping criteria
        self.stats = {'max_val_acc': 0,
                      'max_val_epoch': -1,
                      'first_epoch': 1,
                      'train_loss': [],
                      'val_acc': [],
                      'val_loss': [],
                      'runtime': []}

    def load_checkpoint(self, devicehandler, checkpointhandler, model, optimizer, scheduler):
        checkpoint = checkpointhandler.load(self.checkpoint_file, devicehandler.get_device(), model,
                                            optimizer, scheduler)
        self.stats = checkpoint['stats']
        self.first_epoch = checkpoint['next_epoch']

    def train_model(self, train_loader, model, criterion_func, devicehandler, optimizer):

        logging.info('Training model...')
        train_loss = 0

        # Set model in 'Training mode'
        model.train()

        # process mini-batches
        for i, (inputs, targets) in enumerate(train_loader):
            # prep
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            inputs, targets = devicehandler.move_data_to_device(model, inputs, targets)

            # compute forward pass
            out = model.forward(inputs)

            # calculate loss
            loss = criterion_func(out, targets)
            train_loss += loss.item()

            # compute backward pass
            loss.backward()

            # update model weights
            optimizer.step()

            # delete mini-batch data from device
            del inputs
            del targets

        # calculate average loss across all mini-batches
        train_loss /= len(train_loader)

        logging.info('Training iteration is finished.')
        return train_loss

    def evaluate_model(self, val_loader, model, criterion_func, devicehandler, hit_func):

        logging.info('Evaluating model...')
        val_loss = 0
        hits = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for (inputs, targets) in val_loader:
                # prep
                inputs, targets = devicehandler.move_data_to_device(model, inputs, targets)

                # forward pass
                out = model.forward(inputs)

                # calculate validation loss
                loss = criterion_func(out, targets)
                val_loss += loss.item()

                # calculate number of accurate predictions for this batch
                out = out.cpu().detach().numpy()  # extract from gpu
                hits += hit_func(out, targets)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics
            val_loss /= len(val_loader)  # average per mini-batch
            val_acc = hits / len(val_loader.dataset)  # global accuracy

            logging.info('Evaluation iteration is finished.')
            return val_loss, val_acc

    def test_model(self, test_loader, model, devicehandler):

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
                inputs, targets = devicehandler.move_data_to_device(model, inputs, targets)

                # forward pass
                out = model.forward(inputs)

                # capture output for mini-batch
                out = out.cpu().detach().numpy()  # extract from gpu
                output.append(out)

        logging.info('Testing is finished.')
        return np.concatenate(output, axis=0)

    def stopping_criteria_is_met(self):
        logging.info('Checking early stopping criteria...')

        stopping_criteria_met = False

        if stopping_criteria_met:
            logging.info('Early stopping criteria is met.')
        else:
            logging.info('Early stopping criteria is not met.')

        return stopping_criteria_met

    def report_stats(self, wandbconnector):

        # update best model
        updates_dict = {'best_epoch': self.stats['max_val_acc_epoch'],
                        'best_accuracy': self.stats['max_val_acc']}
        wandbconnector.update_best_model(updates_dict)

        # build stats dictionary of current epoch only
        epoch_stats = {'epoch': self.stats['epoch'][-1],
                       'train_loss': self.stats['train_loss'][-1],
                       'val_loss': self.stats['val_loss'][-1],
                       'val_acc': self.stats['val_acc'][-1],
                       'runtime': self.stats['runtime'][-1]}

        # save epoch stats to wandb
        wandbconnector.log_stats(epoch_stats)
        logging.info(epoch_stats)

    def collect_stats(self, epoch, train_loss, val_loss, val_acc, start, end):

        # calculate runtime
        runtime = end - start

        # update stats
        self.stats['runtime'].append(runtime)
        self.stats['epoch'].append(epoch)
        self.stats['train_loss'].append(train_loss)
        self.stats['val_loss'].append(val_loss)
        self.stats['val_acc'].append(val_acc)

        if val_acc > self.stats['max_val_acc']:
            self.stats['max_val_acc'] = val_acc
            self.stats['max_val_acc_epoch'] = epoch

    def run_training_epochs(self,
                            train_loader,
                            val_loader,
                            model,
                            optimizer,
                            scheduler,
                            criterion_func,
                            hit_func,
                            devicehandler,
                            checkpointhandler,
                            schedulerhandler,
                            wandbconnector):

        # load checkpoint if necessary
        if self.load_from_checkpoint:
            self.load_checkpoint(devicehandler, checkpointhandler, model, optimizer, scheduler)

        # run epochs
        for epoch in range(self.first_epoch, self.num_epochs + 1):
            # record start time
            start = time.time()

            # train
            train_loss = self.train_model(train_loader, model, criterion_func, devicehandler, optimizer)

            # validate
            val_loss, val_acc = self.evaluate_model(val_loader, model, criterion_func, devicehandler, hit_func)

            # stats
            end = time.time()
            self.collect_stats(epoch, train_loss, val_loss, val_acc, start, end)
            self.report_stats(wandbconnector)

            # scheduler
            schedulerhandler.update_scheduler(scheduler, val_acc)

            # save model in case of issues
            checkpointhandler.save(model, optimizer, scheduler, epoch + 1, self.stats)

            # check if early stopping criteria is met
            if self.stopping_criteria_is_met():
                break  # stop running epochs
