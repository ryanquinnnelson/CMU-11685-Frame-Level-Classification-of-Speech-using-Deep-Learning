import configparser
from octopus.octopus import Octopus

config = configparser.ConfigParser()

# default
config['DEFAULT'] = {}
config['DEFAULT']['run_name'] = 'Run-07'

# debug
config['debug'] = {}
config['debug']['debug_file'] = '/Users/ryanqnelson/Desktop/debug.log'

# kaggle
config['kaggle'] = {}
config['kaggle']['download_from_kaggle'] = 'True'
config['kaggle']['kaggle_dir'] = '/Users/ryanqnelson/Desktop/test/.kaggle'
config['kaggle']['token_file'] = '/Users/ryanqnelson/Desktop/test/kaggle.json'
config['kaggle']['content_dir'] = '/Users/ryanqnelson/Desktop/test/content/'
config['kaggle']['competition'] = 'hw1p2-toy-problem'
config['kaggle']['delete_zipfiles_after_unzipping'] = 'True'

# wandb
config['wandb'] = {}
config['wandb']['entity'] = 'ryanquinnnelson'
config['wandb']['project'] = 'CMU-11685-HW1P2-octopus-4'
config['wandb']['notes'] = 'Simple MLP'
config['wandb']['tags'] = 'MLP,octopus'

# stats
config['stats'] = {}
config['stats']['comparison_metric'] = 'val_loss'
config['stats']['comparison_best_is_max'] = 'False'
config['stats']['comparison_patience'] = '20'
config['stats']['val_metric_name'] = 'val_acc'

# data
config['data'] = {}
config['data']['data_dir'] = '/Users/ryanqnelson/Desktop/test/content/competitions/hw1p2-toy-problem'
config['data']['output_dir'] = '/Users/ryanqnelson/Desktop/test/output'
config['data']['train_data_file'] = 'toy_train_data.npy'
config['data']['train_label_file'] = 'toy_train_label.npy'
config['data']['val_data_file'] = 'toy_val_data.npy'
config['data']['val_label_file'] = 'toy_val_label.npy'
config['data']['test_data_file'] = 'toy_test_data.npy'
config['data']['input_size'] = '200'
config['data']['output_size'] = '71'

# model
config['model'] = {}
config['model']['model_type'] = 'MLP'

# checkpoints
config['checkpoint'] = {}
config['checkpoint']['checkpoint_dir'] = '/Users/ryanqnelson/Desktop/test/checkpoints'
config['checkpoint']['delete_existing_checkpoints'] = 'True'
config['checkpoint']['load_from_checkpoint'] = 'False'
config['checkpoint']['checkpoint_file'] = 'None'

# hyperparameters
config['hyperparameters'] = {}
config['hyperparameters']['dataloader_num_workers'] = '8'
config['hyperparameters']['dataloader_pin_memory'] = 'True'
config['hyperparameters']['dataloader_batch_size'] = '128'
config['hyperparameters']['dataset_kwargs'] = 'context=2'
config['hyperparameters']['num_epochs'] = '30'
config['hyperparameters']['hidden_layer_sizes'] = '128'
config['hyperparameters']['dropout_rate'] = '0.0'
config['hyperparameters']['batch_norm'] = 'False'
config['hyperparameters']['activation_func'] = 'ReLU'
config['hyperparameters']['criterion_type'] = 'CrossEntropyLoss'
config['hyperparameters']['optimizer_type'] = 'Adam'
config['hyperparameters']['optimizer_kwargs'] = 'lr=0.001'
config['hyperparameters']['scheduler_type'] = 'ReduceLROnPlateau'
config['hyperparameters']['scheduler_kwargs'] = 'factor=0.1,patience=5,mode=max,verbose=True'
config['hyperparameters']['scheduler_plateau_metric'] = 'val_acc'


def test_Octopus__init___use_kaggle():
    config['kaggle']['download_from_kaggle'] = 'True'

    octopus = Octopus(config)
    assert octopus.kaggleconnector is not None


def test_Octopus__init___dont_use_kaggle():
    config['kaggle']['download_from_kaggle'] = 'False'
    octopus = Octopus(config)
    assert octopus.kaggleconnector is None
