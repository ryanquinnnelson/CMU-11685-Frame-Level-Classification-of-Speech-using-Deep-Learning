import logging
from collections import OrderedDict
import torch.nn as nn


def _get_activation_function(activation_func):
    act = None

    if activation_func == 'ReLU':
        act = nn.ReLU(inplace=True)

    return act


def _build_sequence(sizes, num_hidden_layers, activation_func, dropout_rate, batch_norm):
    sequence = []
    for i in range(num_hidden_layers + 1):  # add final layer for output

        linear_tuple = ('lin' + str(i + 1), nn.Linear(sizes[i], sizes[i + 1]))
        sequence.append(linear_tuple)

        # add the following after each hidden linear layer
        if i < num_hidden_layers:  # not final layer

            if batch_norm:
                batch_norm_tuple = ('bn' + str(i + 1),
                                    nn.BatchNorm1d(sizes[i + 1]))
                sequence.append(batch_norm_tuple)

            if dropout_rate > 0:
                dropout_tuple = ('drop' + str(i + 1),
                                 nn.Dropout(dropout_rate))
                sequence.append(dropout_tuple)

            # activation layer
            activation_tuple = (activation_func + str(i + 1),
                                _get_activation_function(activation_func))
            sequence.append(activation_tuple)

    return sequence


class MLP(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers, hidden_layer_sizes, activation_func, dropout_rate,
                 batch_norm):
        super(MLP, self).__init__()

        logging.info('Initializing model...')

        # set layer sizes
        sizes = [input_size] + hidden_layer_sizes + [output_size]

        # build layer tuples based on parameters
        sequence = _build_sequence(sizes, num_hidden_layers, activation_func, dropout_rate, batch_norm)

        # build Sequential model from sequence
        self.model = nn.Sequential(OrderedDict(sequence))

        logging.info('Model initialized.')
        logging.info(f'\n{self.model}')

    def forward(self, x):
        return self.model(x)  # self.layers(x) also works

# # setup model
# INPUT_SIZE = (2 * CONFIG['context'] + 1) * 40
# OUTPUT_SIZE = 71
# model = mb.MLP(INPUT_SIZE,
#                OUTPUT_SIZE,
#                CONFIG['num_hidden_layers'],
#                CONFIG,
#                dropout_rate=CONFIG['dropout_rate'],
#                batch_norm=CONFIG['batch_norm'])
# print(model)
#
# # move model to device before initializing optimizer
# # https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
# if device.type == 'cuda':
#     model = model.to(torch.device('cuda'))
