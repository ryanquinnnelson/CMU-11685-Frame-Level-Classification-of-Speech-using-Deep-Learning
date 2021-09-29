"""
All things related to wandb.
"""
import logging
import subprocess

import pandas as pd


class WandbConnector:
    def __init__(self, entity, run_name, project, notes, tags, config):
        self.entity = entity
        self.run_name = run_name
        self.project = project
        self.notes = notes
        self.tags = tags
        self.config = config
        self.wandb_config = None

    def setup(self):
        logging.info('Setting up wandb connector...')

        _install()
        _login()
        self.wandb_config = _initialize(self.run_name, self.project, self.notes, self.tags, self.config)

    def watch(self, model):
        import wandb
        wandb.watch(model)  # log the network weight histograms

    def log_stats(self, stats_dict):
        import wandb
        wandb.log(stats_dict)

    def update_best_model(self, updates_dict):
        import wandb

        for key, value in updates_dict.items():
            wandb.run.summary[key] = value

    def _concatenate_run_metrics(self, metric, epoch):
        runs = self._pull_runs()
        valid_run_metrics = []
        for run in runs:
            name = run.name
            metrics_df = run.history()
            cols = list(metrics_df.columns)

            if not metrics_df.empty and 'epoch' in cols and metric in cols:
                # add run name to columns
                metrics_df['name'] = name
                valid_run_metrics.append(metrics_df[['name', 'epoch', metric]])

        # pool metrics for all runs
        if len(valid_run_metrics) > 0:
            df = pd.concat(valid_run_metrics)
            df = df[df['epoch'] == epoch]
        else:
            df = pd.DataFrame(columns=['name', 'epoch', metric])  # empty dataframe

        return df

    def get_best_value(self, metric, epoch, best_is_max):

        # gather the metrics for each valid run
        metrics_df = self._concatenate_run_metrics(metric, epoch)

        # temp output
        a = metrics_df[metrics_df['epoch'] == epoch].sort_values(metric)
        logging.info(f'Gathered metrics for epoch {epoch} from wandb:\n{a}')

        if metrics_df.empty:
            best_name, best_val = None, None
        else:
            best_name, best_val = _calculate_best_value(metrics_df, metric, epoch, best_is_max)

        return best_name, best_val

    def _pull_runs(self):
        import wandb
        api = wandb.Api()

        runs = api.runs(f'{self.entity}/{self.project}')
        return runs


def _calculate_best_value(metrics_df, metric, epoch, best_is_max):
    if best_is_max:
        # find the max value of the metric for each epoch
        bests = metrics_df.groupby(by='epoch').max().reset_index()
    else:
        # find the min
        bests = metrics_df.groupby(by='epoch').min().reset_index()

    # choose the best value for the epoch we care about
    best_val = bests[bests['epoch'] == epoch][metric].item()
    best_name = metrics_df[metrics_df[metric] == best_val]['name'].iloc[0]
    return best_name, best_val


def _install():
    logging.info('Installing wandb...')

    process = subprocess.Popen(['pip', 'install', '--upgrade', 'wandb==0.10.8'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))


def _login():
    logging.info('Logging into wandb...')

    import wandb
    wandb.login()


def _initialize(name, project, notes, tags, config):
    logging.info('Initializing wandb...')

    import wandb
    wandb.init(name=name,
               project=project,
               notes=notes,
               tags=tags,
               config=config)

    return wandb.config
