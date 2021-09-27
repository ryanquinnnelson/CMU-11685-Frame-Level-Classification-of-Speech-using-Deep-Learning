import logging
import subprocess


def setup(name, project, notes, tags, config):
    logging.info('Setting up wandb...')
    _install()
    _login()
    wandb_config = _initialize(name, project, notes, tags, config)
    logging.info('wandb is set up.')

    return wandb_config


def _install():
    logging.info('Installing wandb...')

    process = subprocess.Popen(['pip', 'install', '--upgrade', 'wandb==0.10.8'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))
    logging.info('wandb is installed.')


def _login():
    logging.info('Logging into wandb...')

    import wandb
    wandb.login()
    logging.info('Logged into wandb.')


def _initialize(name, project, notes, tags, config):
    logging.info('Initializing wandb...')

    import wandb
    wandb.init(name=name,
               project=project,
               notes=notes,
               tags=tags,
               config=config)

    logging.info('wandb is initialized.')

    return wandb.config


def watch(model):
    import wandb
    wandb.watch(model)


def log_stats(stats):
    import wandb
    wandb.log(stats)


def update_best_model(epoch, val_acc):
    import wandb
    wandb.run.summary['best_accuracy'] = val_acc
    wandb.run.summary['best_epoch'] = epoch
