import logging
import torch


class Evaluation:
    def __init__(self, model, val_loader, criterion_func, evaluate_batch_func, evaluate_epoch_func, devicehandler,
                 num_epochs):
        self.model = model
        self.val_loader = val_loader
        self.criterion_func = criterion_func
        self.evaluate_batch_func = evaluate_batch_func
        self.evaluate_epoch_func = evaluate_epoch_func
        self.devicehandler = devicehandler
        self.num_epochs = num_epochs

    def evaluate_model(self, epoch):
        logging.info(f'Running epoch {epoch}/{self.num_epochs} of evaluation...')
        val_loss = 0
        val_metric = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            self.model.eval()

            # process mini-batches
            for (inputs, targets) in self.val_loader:
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(self.model, inputs, targets)

                # forward pass
                out = self.model.forward(inputs)

                # calculate validation loss
                loss = self.criterion_func(out, targets)
                val_loss += loss.item()

                # calculate number of accurate predictions for this batch
                out = out.cpu().detach().numpy()  # extract from gpu
                val_metric += self.evaluate_batch_func(out, targets)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate evaluation metrics
            val_loss /= len(self.val_loader)  # average per mini-batch
            val_metric = self.evaluate_epoch_func(val_metric, len(self.val_loader), len(self.val_loader.dataset))

            return val_loss, val_metric
