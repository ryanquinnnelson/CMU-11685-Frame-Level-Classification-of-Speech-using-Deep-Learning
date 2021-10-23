"""
All things related to schedulers.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch.optim as optim

"""
StepLR: Decays the learning rate of each parameter group by gamma every step_size epochs
- step_size
- gamma

MultiStepLR: Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones
- milestones
- gamma

ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving
- factor
- patience
- mode (Note: if we want the metrics to be large (i.e. accuracy), we should set mode='max')
"""


class SchedulerHandler:

    def __init__(self, scheduler_type, scheduler_kwargs, scheduler_plateau_metric):
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_plateau_metric = scheduler_plateau_metric

    def get_scheduler(self, optimizer):
        scheduler = None

        if self.scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **self.scheduler_kwargs)

        elif self.scheduler_type == 'MultiStepLR':

            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **self.scheduler_kwargs)

        elif self.scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_kwargs)

        logging.info(f'Scheduler initialized:\n{scheduler}\n{scheduler.state_dict()}')
        return scheduler

    def update_scheduler(self, scheduler, stats):
        if self.scheduler_type == 'ReduceLROnPlateau':
            metric_val = stats[self.scheduler_plateau_metric][-1]
            scheduler.step(metric_val)
        else:
            scheduler.step()
