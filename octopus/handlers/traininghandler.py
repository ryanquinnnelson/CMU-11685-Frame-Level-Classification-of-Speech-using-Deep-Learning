"""
All things related to training models.
"""
__author__ = 'ryanquinnnelson'

import logging
import time

import torch
import numpy as np


def train_model(epoch, num_epochs, train_loader, model, criterion_func, devicehandler, optimizer):
    logging.info(f'Running epoch {epoch}/{num_epochs} of training...')
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

    return train_loss




def evaluate_model(epoch, num_epochs, val_loader, model, loss_func, evaluate_batch_func, evaluate_epoch_func,
                   devicehandler):
    logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')
    val_loss = 0
    eval_metric = 0

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
            loss = loss_func(out, targets)
            val_loss += loss.item()

            # calculate number of accurate predictions for this batch
            out = out.cpu().detach().numpy()  # extract from gpu
            eval_metric += evaluate_batch_func(out, targets)

            # delete mini-batch from device
            del inputs
            del targets

        # calculate evaluation metrics
        val_loss /= len(val_loader)  # average per mini-batch
        eval_metric = evaluate_epoch_func(eval_metric, len(val_loader), len(val_loader.dataset))

        return val_loss, eval_metric


def test_model(epoch, num_epochs, test_loader, model, devicehandler):
    logging.info(f'Running epoch {epoch}/{num_epochs} of testing...')
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

    return np.concatenate(output, axis=0)


class TrainingHandler:

    def __init__(self, load_from_checkpoint, first_epoch, num_epochs, comparison_metric, comparison_best_is_max,
                 comparison_patience,
                 checkpoint_file=None):
        self.load_from_checkpoint = load_from_checkpoint
        self.checkpoint_file = checkpoint_file
        self.first_epoch = first_epoch
        self.num_epochs = num_epochs
        self.comparison_metric = comparison_metric
        self.best_is_max = comparison_best_is_max
        self.comparison_patience = comparison_patience
        self.num_epochs_worse_than_best_model = 0

        # setup tracking variables and early stopping criteria
        self.stats = {'max_val_acc': -1.0,
                      'max_val_acc_epoch': -1,
                      'train_loss': [],
                      'val_acc': [],
                      'val_loss': [],
                      'runtime': [],
                      'epoch': []}

    def _load_checkpoint(self, devicehandler, checkpointhandler, model, optimizer, scheduler):
        device = devicehandler.get_device()
        checkpoint = checkpointhandler.load(self.checkpoint_file, device, model, optimizer, scheduler)

        # restore stats
        self.stats = checkpoint['stats']

        # set which epoch to start from
        self.first_epoch = checkpoint['next_epoch']

    def _model_is_worse_by_comparison_metric(self, epoch, wandbconnector):

        # check whether the metric we are using for comparison against other runs
        # is better than other runs for this epoch
        logging.info('Checking whether comparison metric for current model is worse than the best model so far...')
        best_name, best_val = wandbconnector.get_best_value(self.comparison_metric, epoch, self.best_is_max)
        model_name = wandbconnector.run_name
        model_val = self.stats[self.comparison_metric][-1]
        print(f'best:\t{best_name}\t{best_val}\nmodel:\t{model_name}\t{model_val}')

        if best_val is not None and self.best_is_max:
            # compare values for this epoch
            if model_val < best_val:
                model_is_worse = True
                logging.info('A previous model has the best value for the comparison metric for this epoch.')
            else:
                logging.info('Current model has the best value for the comparison metric for this epoch.')
                model_is_worse = False
        elif best_val is not None and not self.best_is_max:  # best is min
            # compare values for this epoch
            if model_val > best_val:
                model_is_worse = True
                logging.info('A previous model has the best value for the comparison metric for this epoch.')
            else:
                logging.info('Current model has the best value for the comparison metric for this epoch.')
                model_is_worse = False
        else:
            # no model to compare against for this epoch
            model_is_worse = False

        return model_is_worse

    def _stopping_criteria_is_met(self, epoch, wandbconnector):
        logging.info('Checking early stopping criteria...')

        # criteria 1 - metric comparison
        model_is_worse = self._model_is_worse_by_comparison_metric(epoch, wandbconnector)
        if model_is_worse:
            self.num_epochs_worse_than_best_model += 1
            logging.info('Number of epochs in a row this model is worse than best model' +
                         f':{self.num_epochs_worse_than_best_model}')
            logging.info('Number of epochs in a row this model can be worse than best model ' +
                         f'before stopping:{self.comparison_patience}')
        else:
            self.num_epochs_worse_than_best_model = 0  # reset value

        # check all criteria to determine if we need to stop learning
        if self.num_epochs_worse_than_best_model > self.comparison_patience:
            stopping_criteria_met = True
            logging.info('Early stopping criteria is met.')
        else:
            stopping_criteria_met = False
            logging.info('Early stopping criteria is not met.')

        return stopping_criteria_met

    def _report_stats(self, wandbconnector):

        # update best model
        updates_dict = {'best_epoch': self.stats['max_val_acc_epoch'],
                        'best_accuracy': self.stats['max_val_acc']}
        wandbconnector.update_best_model(updates_dict)

        # save epoch stats to wandb
        epoch_stats = {'epoch': self.stats['epoch'][-1],
                       'train_loss': self.stats['train_loss'][-1],
                       'val_loss': self.stats['val_loss'][-1],
                       'val_acc': self.stats['val_acc'][-1],
                       'runtime': self.stats['runtime'][-1]}
        wandbconnector.log_stats(epoch_stats)
        logging.info(f'stats:{epoch_stats}')

    def _collect_stats(self, epoch, train_loss, val_loss, val_acc, start, end):

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

    def run_training_epochs(self, train_loader, val_loader, test_loader, model, optimizer, scheduler, loss_func,
                            evaluate_batch_func, evaluate_epoch_func, format_func,
                            datahandler, devicehandler, checkpointhandler, schedulerhandler, wandbconnector):

        # load checkpoint if necessary
        if self.load_from_checkpoint:
            self._load_checkpoint(devicehandler, checkpointhandler, model, optimizer, scheduler)

        # run epochs
        for epoch in range(self.first_epoch, self.num_epochs + 1):
            # record start time
            start = time.time()

            # train
            train_loss = train_model(epoch, self.num_epochs, train_loader, model, loss_func, devicehandler, optimizer)

            # validate
            val_loss, val_acc = evaluate_model(epoch, self.num_epochs, val_loader, model, loss_func,
                                               evaluate_batch_func, evaluate_epoch_func, devicehandler)

            # test
            out = test_model(epoch, self.num_epochs, test_loader, model, devicehandler)
            df = format_func(out)
            datahandler.save(df, epoch)

            # stats
            end = time.time()
            self._collect_stats(epoch, train_loss, val_loss, val_acc, start, end)
            self._report_stats(wandbconnector)

            # scheduler
            schedulerhandler.update_scheduler(scheduler, val_acc)

            # save model checkpoint
            checkpointhandler.save(model, optimizer, scheduler, epoch + 1, self.stats)

            # check if early stopping criteria is met
            if self._stopping_criteria_is_met(epoch, wandbconnector):
                logging.info('Early stopping criteria is met. Stopping the training process...')
                break  # stop running epochs
