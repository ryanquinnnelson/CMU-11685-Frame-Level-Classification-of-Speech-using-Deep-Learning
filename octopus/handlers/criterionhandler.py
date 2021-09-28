"""
All things related to criterion.
"""
import logging

import torch.nn as nn


class CriterionHandler:

    def __init__(self, criterion_type):
        self.criterion_type = criterion_type

    def get_loss_function(self):
        criterion = None
        if self.criterion_type == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()

        logging.info(f'Criterion is set:{criterion}.')
        return criterion
