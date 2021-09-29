import logging
import torch
import numpy as np


class Testing:
    def __init__(self, model, test_loader, devicehandler, num_epochs):
        self.model = model
        self.test_loader = test_loader
        self.devicehandler = devicehandler
        self.num_epochs = num_epochs

    def test_model(self, epoch):
        logging.info(f'Running epoch {epoch}/{self.num_epochs} of testing...')
        output = []

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            self.model.eval()

            # process mini-batches
            for batch in self.test_loader:

                if type(batch) is tuple:
                    # loader contains inputs and targets
                    inputs = batch[0]
                    targets = batch[1]
                else:
                    # loader contains only inputs
                    inputs = batch
                    targets = None

                # prep
                inputs, targets = self.devicehandler.move_data_to_device(self.model, inputs, targets)

                # forward pass
                out = self.model.forward(inputs)

                # capture output for mini-batch
                out = out.cpu().detach().numpy()  # extract from gpu if necessary
                output.append(out)

        return np.concatenate(output, axis=0)
