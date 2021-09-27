import configparser
import sys
import logging
import os
import time

# local modules
import pipeline.kaggleconnector as kaggleconnector
import pipeline.wandbconnector as wandbconnector
import pipeline.datadealer as datadealer
import pipeline.devicedealer as devicedealer
import pipeline.checkpointer as checkpointer
import pipeline.modeler as modeler
import pipeline.componentdealer as componentdealer
import pipeline.trainer as trainer
import customized.datasets as datasets


def main():
    # parse config file
    config_path = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_path)

    # 1 - setup
    _setup_logging(config)
    _setup_kaggle(config)
    _setup_wandb(config)
    _setup_checkpoint_directory(config)
    device = devicedealer.setup_device()
    model = _setup_and_move_model(config, device)
    criterion_func, optimizer, scheduler = _setup_training_components(config, model)

    # load data
    train_loader, val_loader, test_loader = _load_data(config, device)

    # 2 - training
    # setup tracking variables and early stopping criteria
    stats = {'max_val_acc': 0, 'first_epoch': 1, 'train_loss': [], 'val_acc': [], 'val_loss': [], 'runtime': []}
    _load_checkpoint_if_necessary(model, optimizer, scheduler, config, stats, device)

    # run epochs
    for epoch in range(stats['first_epoch'], config['hyperparameters'].getint('num_epochs') + 1):
        # record start time
        start = time.time()

        # train
        train_loss = trainer.train_model(train_loader, model, optimizer, criterion_func, device)

        # validate
        val_acc, val_loss = trainer.evaluate_model(val_loader,
                                                   model,
                                                   criterion_func,
                                                   device,
                                                   datasets.calculate_n_hits)

        # collect stats
        end = time.time()
        _stats_collection(stats, epoch, train_loss, val_acc, val_loss, start, end)

        # update scheduler
        componentdealer.update_scheduler(scheduler,
                                         config['hyperparameters']['scheduler_type'],
                                         config['hyperparameters']['scheduler_plateau_metric'],
                                         stats)

        # save model in case of issues
        checkpointer.save_checkpoint(model,
                                     optimizer,
                                     scheduler,
                                     epoch + 1,
                                     stats['max_val_acc'],
                                     config['checkpoint']['checkpoint_dir'],
                                     config['wandb']['name'])

        # check if early stopping criteria is met
        if trainer.perform_early_stop(stats):
            break  # stop running epochs

    # 3 - test model
    out = trainer.test_model(test_loader, model, device)
    predictions = datasets.convert_to_class_labels(out)
    trainer.save_results(predictions,
                         config['wandb']['name'],
                         config['DEFAULT']['results_dir'])


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


def _setup_logging(config):
    debug_file = config['DEFAULT']['debug_file']

    # delete any older debug files if they exist
    if os.path.isfile(debug_file):
        os.remove(debug_file)

    # https://youtrack.jetbrains.com/issue/PY-39762
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(debug_file), logging.StreamHandler(sys.stdout)]
                        )


def _setup_kaggle(config):
    # kaggle
    kaggleconnector.setup(config['kaggle']['kaggle_dir'],
                          config['kaggle']['content_dir'],
                          config['kaggle']['token_file'])

    # competition files
    competition_path = kaggleconnector.get_competition_path(config['kaggle']['content_dir'],
                                                            config['kaggle']['competition'])
    if not os.path.isdir(competition_path):
        kaggleconnector.download(config['kaggle']['competition'])
        kaggleconnector.unzip(competition_path)


def _setup_wandb(config):
    # wandb
    config = wandbconnector.setup(config['wandb']['name'],
                                  config['wandb']['project'],
                                  config['wandb']['notes'],
                                  config['wandb']['tags'],
                                  dict(config['hyperparameters']))

    return config


def _setup_checkpoint_directory(config):
    # setup checkpoint directory
    checkpointer.setup_checkpoint_directory(config['checkpoint']['checkpoint_dir'],
                                            config['checkpoint']['delete_existing_checkpoints'])


def _load_data(config, device):
    competition_path = kaggleconnector.get_competition_path(config['kaggle']['content_dir'],
                                                            config['kaggle']['competition'])

    dataset_dict = _to_dict(config['data']['dataset_kwargs'])
    train_loader, val_loader, test_loader = datadealer.load(competition_path,
                                                            config['data']['train_data_file'],
                                                            config['data']['train_label_file'],
                                                            config['data']['val_data_file'],
                                                            config['data']['val_label_file'],
                                                            config['data']['test_data_file'],
                                                            datasets.TrainValDataset,
                                                            datasets.TestDataset,
                                                            device,
                                                            config['hyperparameters'].getint('batch_size'),
                                                            config['data'].getint('num_workers'),
                                                            config['data'].getboolean('pin_memory'),
                                                            **dataset_dict)

    return train_loader, val_loader, test_loader


def _setup_and_move_model(config, device):
    model = modeler.MLP(config['data'].getint('input_size'),
                       config['data'].getint('output_size'),
                       config['hyperparameters'].getint('num_hidden_layers'),
                       _to_int_list(config['hyperparameters']['hidden_layer_sizes']),
                       config['hyperparameters']['activation_func'],
                       config['hyperparameters'].getfloat('dropout_rate'),
                       config['hyperparameters'].getboolean('batch_norm'))

    # move model to device before initializing optimizer
    # https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least
    devicedealer.move_model_to_device(device, model)

    # watch model
    wandbconnector.watch(model)  # log the network weight histograms

    return model


def _setup_training_components(config, model):
    # loss function
    criterion = componentdealer.set_criterion(config['hyperparameters']['criterion_type'])

    # optimizer
    optimizer_dict = _to_dict(config['hyperparameters']['optimizer_kwargs'])
    optimizer = componentdealer.set_optimizer(config['hyperparameters']['optimizer_type'],
                                         model,
                                         **optimizer_dict)

    # scheduler
    scheduler_dict = _to_dict(config['hyperparameters']['scheduler_kwargs'])
    scheduler = componentdealer.set_scheduler(config['hyperparameters']['scheduler_type'],
                                         optimizer,
                                         **scheduler_dict)

    return criterion, optimizer, scheduler


def _load_checkpoint_if_necessary(model, optimizer, scheduler, config, stats, device):
    if config['checkpoint'].getboolean('load_from_checkpoint'):
        checkpoint = checkpointer.load_checkpoint(model,
                                                  optimizer,
                                                  scheduler,
                                                  config['checkpoint']['checkpoint_file'],
                                                  device)
        stats['first_epoch'] = checkpoint['next_epoch']
        stats['max_val_acc'] = checkpoint['max_val_acc']


def _append_stats(stats, key, val):
    stats[key].append(val)


def _update_stats(stats, key, val):
    if val > stats[key]:
        stats[key] = val


def _stats_collection(stats, epoch, train_loss, val_acc, val_loss, start, end):
    # calculate runtime
    runtime = end - start

    # save model to wandb if it is better than previous versions
    if val_acc > stats['max_val_acc']:
        wandbconnector.update_best_model(epoch, val_acc)

    # update stats
    _append_stats(stats, 'val_acc', val_acc)
    _append_stats(stats, 'val_loss', val_loss)
    _append_stats(stats, 'train_loss', train_loss)
    _append_stats(stats, 'runtime', runtime)
    _update_stats(stats, 'max_val_acc', val_acc)

    # build stats dictionary of current epoch only
    epoch_stats = {'val_acc': stats['val_acc'][-1],
                   'val_loss': stats['val_loss'][-1],
                   'train_loss': stats['train_loss'][-1],
                   'runtime': stats['runtime'][-1],
                   'epoch': epoch}

    # save epoch stats to wandb
    wandbconnector.log_stats(epoch_stats)
    logging.info(epoch_stats)


if __name__ == "__main__":
    # execute only if run as a script
    main()
