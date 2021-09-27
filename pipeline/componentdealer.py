import torch.nn as nn
import torch.optim as optim


def set_criterion(criterion_type):
    c = None
    if criterion_type == 'CrossEntropyLoss':
        c = nn.CrossEntropyLoss()
    return c


def set_optimizer(optimizer_type, model, **kwargs):
    o = None
    if optimizer_type == 'Adam':
        o = optim.Adam(model.parameters(), **kwargs)

    return o


def set_scheduler(scheduler_type, optimizer, **kwargs):
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

    :param scheduler_type:
    :param optimizer:
    :param config:
    :return:
    """
    scheduler = None

    if scheduler_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, **kwargs)

    elif scheduler_type == 'MultiStepLR':

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)

    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    return scheduler


def update_scheduler(scheduler, scheduler_type, plateau_metric, stats):
    if scheduler_type == 'ReduceLROnPlateau':
        metric = stats[plateau_metric][-1]  # latest value
        scheduler.step(metric)
    else:
        scheduler.step()
