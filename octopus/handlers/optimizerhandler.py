"""
All things related to optimizers.
"""

import torch.optim as optim


class OptimizerHandler:

    def __init__(self, optimizer_type, optimizer_kwargs):
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs

    def get_optimizer(self, model):
        opt = None
        if self.optimizer_type == 'Adam':
            opt = optim.Adam(model.parameters(), **self.optimizer_kwargs)

        return opt
