"""
All things related to device.
"""

import logging

import torch


class DeviceHandler:
    def __init__(self):
        self.device = None

    def setup(self):
        logging.info('Setting up device handler...')
        logging.info('Checking device...')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda':
            logging.info(f'Gpu detected. device is set to {self.device}.')
        else:
            logging.info(f'No gpu detected. device is set to {self.device}.')

    def get_device(self):
        return self.device

    def move_model_to_device(self, model):
        """
        Avoids duplication issue with moving to device.

        "Note that calling my_tensor.to(device) returns a new copy of my_tensor on GPU. It does NOT overwrite my_tensor.
        Therefore, remember to manually overwrite tensors: my_tensor = my_tensor.to(torch.device('cuda'))."
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        """

        if self.device.type == 'cuda':
            model = model.to(torch.device('cuda'))

        return model

    def move_data_to_device(self, model, inputs, targets=None):
        """
        Avoids duplication issue with moving to device.

        "Note that calling my_tensor.to(device) returns a new copy of my_tensor on GPU. It does NOT overwrite my_tensor.
        Therefore, remember to manually overwrite tensors: my_tensor = my_tensor.to(torch.device('cuda'))."
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict

        """

        # send input and targets to device
        if self.device.type == 'cuda':
            inputs = inputs.to(torch.device('cuda'))

            if targets is not None:
                targets = targets.to(torch.device('cuda'))

        # validate that model and input/targets are on the same device
        assert next(model.parameters()).device == inputs.device

        if targets is not None:
            assert inputs.device == targets.device

        return inputs, targets

    # def check_status(self):
    #
    #     if self.device.type == 'cuda':
    #         # check gpu properties
    #         from pynvml import *
    #         t = torch.cuda.get_device_properties(0).total_memory
    #         r = torch.cuda.memory_reserved(0)
    #         a = torch.cuda.memory_allocated(0)
    #         f = r - a  # free inside reserved
    #
    #         nvmlInit()
    #         h = nvmlDeviceGetHandleByIndex(0)
    #         info = nvmlDeviceGetMemoryInfo(h)
    #         print(f'total    : {info.total}')
    #         print(f'free     : {info.free}')
    #         print(f'used     : {info.used}')