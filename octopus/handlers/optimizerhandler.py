import torch.optim as optim


class OptimizerHandler:

    def __init__(self, optimizer_type, args_dict):
        self.optimizer_type = optimizer_type
        self.args_dict = args_dict

    def get_optimizer(self, model):
        opt = None
        if self.optimizer_type == 'Adam':
            opt = optim.Adam(model.parameters(), **self.args_dict)

        return opt
