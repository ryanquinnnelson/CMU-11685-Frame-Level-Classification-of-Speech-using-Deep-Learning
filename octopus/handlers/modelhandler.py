"""
All things related to models.
"""
__author__ = 'ryanquinnnelson'

import logging

from octopus.models.MLP import MLP


class ModelHandler:
    def __init__(self,
                 model_type,
                 input_size,
                 output_size,
                 hidden_layer_sizes,
                 activation_func,
                 dropout_rate,
                 batch_norm):
        self.model_type = model_type
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

    def get_model(self):
        logging.info('Initializing model...')
        model = None
        if self.model_type == 'MLP':
            model = MLP(self.input_size,
                        self.output_size,
                        self.hidden_layer_sizes,
                        self.activation_func,
                        self.dropout_rate,
                        self.batch_norm)
        logging.info(f'Model initialized:\n{model}')
        return model
