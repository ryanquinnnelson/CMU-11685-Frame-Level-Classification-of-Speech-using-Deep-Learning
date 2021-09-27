"""
All things related to criterion.
"""

import torch.nn as nn


class CriterionHandler:

    def __init__(self, criterion_type):
        self.criterion_type = criterion_type

    def get_criterion(self):
        criterion = None
        if self.criterion_type == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()

        return criterion
