import logging
import os
import sys

# local modules
import octopus.connectors.kaggleconnector as kc
import octopus.connectors.wandbconnector as wc
import octopus.handlers.checkpointhandler as cph
import octopus.handlers.devicehandler as dh
import octopus.handlers.datahandler as dah
import octopus.handlers.modelhandler as mh
import octopus.handlers.criterionhandler as ch
import octopus.handlers.optimizerhandler as oh
import octopus.handlers.schedulerhandler as sh
import octopus.handlers.traininghandler as th
import customized.datasets as datasets


class Octopus:

    def __init__(self, config):
        self.config = config

        # logging
        _setup_logging(config['DEFAULT']['debug_file'])

        # kaggle
        self.kaggleconnector = kc.KaggleConnector(config['kaggle']['kaggle_dir'],
                                                  config['DEFAULT']['data_dir'],
                                                  config['kaggle']['token_file'],
                                                  config['kaggle']['competition'],
                                                  config['kaggle'].getboolean('delete_zipfiles'))

        # wandb
        self.wandbconnector = wc.WandbConnector(config['wandb']['name'],
                                                config['wandb']['project'],
                                                config['wandb']['notes'],
                                                config['wandb']['tags'],
                                                dict(config['hyperparameters']))

        # checkpoints
        self.checkpointhandler = cph.CheckpointHandler(config['checkpoint']['checkpoint_dir'],
                                                       config['checkpoint']['delete_existing_checkpoints'],
                                                       config['wandb']['name'])

        # data
        if config.has_option('data', 'test_label_file'):
            test_label_file = config['data']['test_label_file']
        else:
            test_label_file = None
        self.datahandler = dah.DataHandler(config['wandb']['name'],
                                           config['DEFAULT']['data_dir'],
                                           config['DEFAULT']['output_dir'],
                                           config['data']['train_data_file'],
                                           config['data']['train_label_file'],
                                           config['data']['val_data_file'],
                                           config['data']['val_label_file'],
                                           config['data']['test_data_file'],
                                           test_label_file,
                                           config['hyperparameters'].getint('batch_size'),
                                           config['data'].getint('num_workers'),
                                           config['data'].getboolean('pin_memory'),
                                           _to_dict(config['data']['dataset_kwargs']))

        # device
        self.devicehandler = dh.DeviceHandler()

        # model
        self.modelhandler = mh.ModelHandler(config['model']['model_type'],
                                            config['data'].getint('input_size'),
                                            config['data'].getint('output_size'),
                                            _to_int_list(config['hyperparameters']['hidden_layer_sizes']),
                                            config['hyperparameters']['activation_func'],
                                            config['hyperparameters'].getfloat('dropout_rate'),
                                            config['hyperparameters'].getboolean('batch_norm'))

        # criterion
        self.criterionhandler = ch.CriterionHandler(config['hyperparameters']['criterion_type'])

        # optimizer
        self.optimizerhandler = oh.OptimizerHandler(config['hyperparameters']['optimizer_type'],
                                                    _to_dict(config['hyperparameters']['optimizer_kwargs']), )
        # scheduler
        self.schedulerhandler = sh.SchedulerHandler(config['hyperparameters']['scheduler_type'],
                                                    _to_dict(config['hyperparameters']['scheduler_kwargs']))

        # training
        if config.has_option('checkpoint', 'checkpoint_file'):
            checkpoint_file = config['checkpoint']['checkpoint_file']
        else:
            checkpoint_file = None
        self.traininghandler = th.TrainingHandler(config['checkpoint'].getboolean('load_from_checkpoint'),
                                                  config['hyperparameters'].getint('num_epochs'),
                                                  checkpoint_file)

    def setup_environment(self):
        logging.info('Setting up environment...')

        # kaggle
        self.kaggleconnector.setup()
        self.kaggleconnector.download_and_unzip()

        # wandb
        self.wandbconnector.setup()

        # checkpoint directory
        self.checkpointhandler.setup()

        # output directory
        self.datahandler.setup()

        logging.info('Environment is set up.')

    def run_pipeline(self):
        """
        Note 1:
        Reason behind moving model to device first:
        https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
        """
        logging.info('Running deep learning pipeline...')

        # model
        model = self.modelhandler.get_model()
        self.devicehandler.move_model_to_device(model)  # move model before initializing optimizer - see Note 1
        self.wandbconnector.watch(model)

        # model components
        criterion_func = self.criterionhandler.get_criterion()
        optimizer = self.optimizerhandler.get_optimizer(model)
        scheduler = self.schedulerhandler.get_scheduler(optimizer)

        # data
        train_loader, val_loader, test_loader = self.datahandler.load(datasets.TrainValDataset,
                                                                      datasets.TrainValDataset,
                                                                      datasets.TestDataset,
                                                                      self.devicehandler.get_device())

        # load checkpoint if necessary
        self.traininghandler.load_checkpoint_if_necessary(self.devicehandler,
                                                          self.checkpointhandler,
                                                          model,
                                                          optimizer,
                                                          scheduler)

        # run training epochs
        self.traininghandler.run_training_epochs(train_loader, val_loader, model, optimizer, scheduler, criterion_func,
                                                 datasets.calculate_n_hits, self.devicehandler, self.checkpointhandler,
                                                 self.schedulerhandler, self.wandbconnector)

        # test model
        out = self.traininghandler.test_model(test_loader, model, self.devicehandler)
        predictions = datasets.convert_to_class_labels(out)
        self.datahandler.save(out)

        logging.info('Deep learning pipeline finished running.')


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
    s1 = s.strip().split(',')
    l = [int(a) for a in s1]
    return l


def _setup_logging(debug_file):
    # delete any older debug files if they exist
    if os.path.isfile(debug_file):
        os.remove(debug_file)

    # write to both debug file and stdout
    # https://youtrack.jetbrains.com/issue/PY-39762
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(debug_file), logging.StreamHandler(sys.stdout)]
                        )
