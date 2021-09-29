"""
Performs environment setup for deep learning and runs a deep learning pipeline.
"""

import logging
import os
import sys

# execute before loading torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # better error tracking from gpu

# local modules
from octopus.connectors.kaggleconnector import KaggleConnector
from octopus.connectors.wandbconnector import WandbConnector
from octopus.handlers.checkpointhandler import CheckpointHandler
from octopus.handlers.devicehandler import DeviceHandler
from octopus.handlers.datahandler import DataHandler
from octopus.handlers.modelhandler import ModelHandler
from octopus.handlers.criterionhandler import CriterionHandler
from octopus.handlers.optimizerhandler import OptimizerHandler
from octopus.handlers.schedulerhandler import SchedulerHandler
from octopus.handlers.statshandler import StatsHandler
from octopus.handlers.phasehandler import PhaseHandler
from octopus.phases.training import Training
from octopus.phases.testing import Testing

# customized to this data
from customized.customized import Evaluation, TrainValDataset, TestDataset, OutputFormatter


class Octopus:

    def __init__(self, config):
        # logging
        _setup_logging(config['debug']['debug_file'])
        _draw_logo()
        logging.info('Initializing octopus...')

        # save configuration
        self.config = config

        # kaggle
        self.kaggleconnector = KaggleConnector(config['kaggle']['kaggle_dir'],
                                               config['data']['data_dir'],
                                               config['kaggle']['token_file'],
                                               config['kaggle']['competition'],
                                               config['kaggle'].getboolean('delete_zipfiles'))

        # wandb
        self.wandbconnector = WandbConnector(config['wandb']['entity'],
                                             config['wandb']['name'],
                                             config['wandb']['project'],
                                             config['wandb']['notes'],
                                             _to_string_list(config['wandb']['tags']),
                                             dict(config['hyperparameters']))

        # checkpoints
        self.checkpointhandler = CheckpointHandler(config['checkpoint']['checkpoint_dir'],
                                                   config['checkpoint']['delete_existing_checkpoints'],
                                                   config['wandb']['name'],
                                                   config['checkpoint'].getboolean('load_from_checkpoint'))

        # criterion
        self.criterionhandler = CriterionHandler(config['hyperparameters']['criterion_type'])

        # data
        if config.has_option('data', 'test_label_file'):
            test_label_file = config['data']['test_label_file']
        else:
            test_label_file = None
        self.datahandler = DataHandler(config['wandb']['name'],
                                       self.kaggleconnector.competition_dir,
                                       config['data']['output_dir'],
                                       config['data']['train_data_file'],
                                       config['data']['train_label_file'],
                                       config['data']['val_data_file'],
                                       config['data']['val_label_file'],
                                       config['data']['test_data_file'],
                                       test_label_file,
                                       config['hyperparameters'].getint('dataloader_batch_size'),
                                       config['hyperparameters'].getint('dataloader_num_workers'),
                                       config['hyperparameters'].getboolean('dataloader_pin_memory'),
                                       _to_dict(config['hyperparameters']['dataset_kwargs']))

        # device
        self.devicehandler = DeviceHandler()

        # model
        self.modelhandler = ModelHandler(config['model']['model_type'],
                                         config['data'].getint('input_size'),
                                         config['data'].getint('output_size'),
                                         _to_int_list(config['hyperparameters']['hidden_layer_sizes']),
                                         config['hyperparameters']['activation_func'],
                                         config['hyperparameters'].getfloat('dropout_rate'),
                                         config['hyperparameters'].getboolean('batch_norm'))

        # optimizer
        self.optimizerhandler = OptimizerHandler(config['hyperparameters']['optimizer_type'],
                                                 _to_dict(config['hyperparameters']['optimizer_kwargs']), )

        # scheduler
        self.schedulerhandler = SchedulerHandler(config['hyperparameters']['scheduler_type'],
                                                 _to_dict(config['hyperparameters']['scheduler_kwargs']),
                                                 config['stats']['val_metric_name'])

        # statshandler
        self.statshandler = StatsHandler(config['stats']['val_metric_name'],
                                         config['stats']['comparison_metric'],
                                         config['stats'].getboolean('comparison_best_is_max'),
                                         config['stats'].getint('comparison_patience'))

        # phasehandler
        if config.has_option('checkpoint', 'checkpoint_file'):
            checkpoint_file = config['checkpoint']['checkpoint_file']
        else:
            checkpoint_file = None
        first_epoch = 1
        self.phasehandler = PhaseHandler(first_epoch,
                                         config['hyperparameters'].getint('num_epochs'),
                                         self.datahandler,
                                         self.devicehandler,
                                         self.statshandler,
                                         self.checkpointhandler,
                                         self.schedulerhandler,
                                         self.wandbconnector,
                                         OutputFormatter(),
                                         config['checkpoint'].getboolean('load_from_checkpoint'),
                                         checkpoint_file)

        logging.info('octopus initialization is complete.')

    def setup_environment(self):
        logging.info('octopus is setting up the environment...')

        # wandb
        self.wandbconnector.setup()

        # kaggle
        self.kaggleconnector.setup()

        # checkpoint directory
        self.checkpointhandler.setup()

        # output directory
        self.datahandler.setup()

        # device
        self.devicehandler.setup()

        logging.info('octopus has finished setting up the environment.')

    def download_data(self):
        logging.info('octopus is downloading data...')
        self.kaggleconnector.download_and_unzip()
        logging.info('octopus has finished downloading data.')

    def run_pipeline(self):
        """
        Note 1:
        Reason behind moving model to device first:
        https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
        """
        logging.info('octopus is running the pipeline...')

        # initialize model
        model = self.modelhandler.get_model()
        self.devicehandler.move_model_to_device(model)  # move model before initializing optimizer - see Note 1
        self.wandbconnector.watch(model)

        # initialize model components
        loss_func = self.criterionhandler.get_loss_function()
        optimizer = self.optimizerhandler.get_optimizer(model)
        scheduler = self.schedulerhandler.get_scheduler(optimizer)

        # load data
        train_loader, val_loader, test_loader = self.datahandler.load(TrainValDataset,
                                                                      TrainValDataset,
                                                                      TestDataset,
                                                                      self.devicehandler)

        # load phases
        training = Training(train_loader, loss_func, self.devicehandler)
        evaluation = Evaluation(val_loader, loss_func, self.devicehandler)
        testing = Testing(test_loader, self.devicehandler)

        # run epochs
        self.phasehandler.process_epochs(model, optimizer, scheduler, training, evaluation, testing)

        logging.info('octopus has finished running the pipeline.')

    def cleanup(self):
        logging.info('octopus shutdown complete.')


def _to_dict(s):
    d = dict()

    pairs = s.split(',')
    for p in pairs:
        key, val = p.strip().split('=')

        # try converting the value to a float
        try:
            val = float(val)
        except ValueError:
            pass  # leave as string

        d[key] = val

    return d


def _to_int_list(s):
    return [int(a) for a in s.strip().split(',')]


def _to_string_list(s):
    return s.strip().split(',')


def _setup_logging(debug_file):
    if os.path.isfile(debug_file):
        os.remove(debug_file)  # delete older debug file if it exists

    # write to both debug file and stdout
    # https://youtrack.jetbrains.com/issue/PY-39762
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(debug_file), logging.StreamHandler(sys.stdout)]
                        )


def _draw_logo():
    logging.info('              _---_')
    logging.info('            /       \\')
    logging.info('           |         |')
    logging.info('   _--_    |         |    _--_')
    logging.info('  /__  \\   \\  0   0  /   /  __\\')
    logging.info('     \\  \\   \\       /   /  /')
    logging.info('      \\  -__-       -__-  /')
    logging.info('  |\\   \\    __     __    /   /|')
    logging.info('  | \\___----         ----___/ |')
    logging.info('  \\                           /')
    logging.info('   --___--/    / \\    \\--___--')
    logging.info('         /    /   \\    \\')
    logging.info('   --___-    /     \\    -___--')
    logging.info('   \\_    __-         -__    _/')
    logging.info('     ----               ----')
    logging.info('')
    logging.info('       O  C  T  O  P  U  S')
    logging.info('')
