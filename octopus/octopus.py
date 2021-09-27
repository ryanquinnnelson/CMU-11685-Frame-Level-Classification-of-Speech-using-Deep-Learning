import logging
import os
import sys
import time

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



        logging.info('Deep learning pipeline finished running.')


#
#     wandb.run.summary['best_accuracy'] = val_acc
#
#
# wandb.run.summary['best_epoch'] = epoch
#
#     # 1 - setup


#
#     # 2 - training
#     # setup tracking variables and early stopping criteria
#     stats = {'max_val_acc': 0, 'first_epoch': 1, 'train_loss': [], 'val_acc': [], 'val_loss': [], 'runtime': []}
#     _load_checkpoint_if_necessary(model, optimizer, scheduler, config, stats, device)
#
#     # run epochs
#     for epoch in range(stats['first_epoch'], config['hyperparameters'].getint('num_epochs') + 1):
#         # record start time
#         start = time.time()
#
#         # train
#         train_loss = trainer.train_model(train_loader, model, optimizer, criterion_func, device)
#
#         # validate
#         val_acc, val_loss = trainer.evaluate_model_on_accuracy(val_loader,
#                                                                model,
#                                                                criterion_func,
#                                                                device,
#                                                                datasets.calculate_n_hits)
#
#         # collect stats
#         end = time.time()
#         _stats_collection(stats, epoch, train_loss, val_acc, val_loss, start, end)
#
#         # update scheduler
#         componentdealer.update_scheduler(scheduler,
#                                          config['hyperparameters']['scheduler_type'],
#                                          config['hyperparameters']['scheduler_plateau_metric'][-1],
#                                          stats)
#
#         # save model in case of issues
#         checkpointer.save_checkpoint(model,
#                                      optimizer,
#                                      scheduler,
#                                      epoch + 1,
#                                      stats['max_val_acc'],
#                                      config['checkpoint']['checkpoint_dir'],
#                                      config['wandb']['name'])
#
#         # check if early stopping criteria is met
#         if trainer.perform_early_stop(stats):
#             break  # stop running epochs
#
#     # 3 - test model
#     out = trainer.test_model(test_loader, model, device)
#     predictions = datasets.convert_to_class_labels(out)
#     octopus.datahandler.save(predictions,
#                               config['wandb']['name'],
#                               config['DEFAULT']['results_dir'])
#
#
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

    # # watch model
    # wandbconnector.watch(model)  # log the network weight histograms


def _setup_data(config, device, data_dir):
    dh =

    dataset_dict =

    return train_loader, val_loader, test_loader

#
#
# def _load_checkpoint_if_necessary(model, optimizer, scheduler, config, stats, device):
#     if config['checkpoint'].getboolean('load_from_checkpoint'):
#         checkpoint = checkpointer.load_checkpoint(model,
#                                                   optimizer,
#                                                   scheduler,
#                                                   config['checkpoint']['checkpoint_file'],
#                                                   device)
#         stats['first_epoch'] = checkpoint['next_epoch']
#         stats['max_val_acc'] = checkpoint['max_val_acc']
#
#
# def _append_stats(stats, key, val):
#     stats[key].append(val)
#
#
# def _update_stats(stats, key, val):
#     if val > stats[key]:
#         stats[key] = val
#
#
# def _stats_collection(stats, epoch, train_loss, val_acc, val_loss, start, end):
#     # calculate runtime
#     runtime = end - start
#
#     # save model to wandb if it is better than previous versions
#     if val_acc > stats['max_val_acc']:
#         wandbconnector.update_best_model(epoch, val_acc)
#
#     # update stats
#     _append_stats(stats, 'val_acc', val_acc)
#     _append_stats(stats, 'val_loss', val_loss)
#     _append_stats(stats, 'train_loss', train_loss)
#     _append_stats(stats, 'runtime', runtime)
#     _update_stats(stats, 'max_val_acc', val_acc)
#
#     # build stats dictionary of current epoch only
#     epoch_stats = {'val_acc': stats['val_acc'][-1],
#                    'val_loss': stats['val_loss'][-1],
#                    'train_loss': stats['train_loss'][-1],
#                    'runtime': stats['runtime'][-1],
#                    'epoch': epoch}
#
#     # save epoch stats to wandb
#     wandbconnector.log_stats(epoch_stats)
#     logging.info(epoch_stats)
