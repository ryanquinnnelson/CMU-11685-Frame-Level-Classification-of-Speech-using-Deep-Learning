"""
All things related to wandb.
"""
import logging
import subprocess


class WandbConnector:
    def __init__(self, run_name, project, notes, tags, config):
        self.run_name = run_name
        self.project = project
        self.notes = notes
        self.tags = tags
        self.config = config
        self.wandb_config = None

    def setup(self):
        logging.info('Setting up wandb...')

        _install()
        _login()
        self.wandb_config = _initialize(self.run_name, self.project, self.notes, self.tags, self.config)

        logging.info('wandb is set up.')

    def watch(self, model):
        import wandb
        wandb.watch(model)

    def log_stats(self, stats_dict):
        import wandb
        wandb.log(stats_dict)

    def update_best_model(self, updates_dict):
        import wandb

        for key, value in updates_dict.items():
            wandb.run.summary[key] = value


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
