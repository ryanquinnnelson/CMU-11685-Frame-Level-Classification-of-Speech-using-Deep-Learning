import os
import logging
from torch.utils.data import DataLoader


def load_dataset(data_path,
                 train_data_file,
                 train_label_file,
                 val_data_file,
                 val_label_file,
                 test_data_file,
                 train_val_dataset,
                 test_dataset,
                 **kwargs):
    logging.info('Loading Datasets...')

    # fully qualified file paths
    train_data_file = os.path.join(data_path, train_data_file)
    train_label_file = os.path.join(data_path, train_label_file)
    val_data_file = os.path.join(data_path, val_data_file)
    val_label_file = os.path.join(data_path, val_label_file)
    test_data_file = os.path.join(data_path, test_data_file)

    # load Datasets
    train_data = train_val_dataset(train_data_file, train_label_file, **kwargs)
    val_data = train_val_dataset(val_data_file, val_label_file, **kwargs)
    test_data = test_dataset(test_data_file, **kwargs)

    logging.info(f'Loaded {train_data.length} records as training data.')
    logging.info(f'Loaded {val_data.length} records as validation data.')
    logging.info(f'Loaded {test_data.length} records as test data.')

    logging.info('Datasets loaded.')
    return train_data, val_data, test_data


def load_dataloaders(train_data, val_data, test_data, device, batch_size, num_workers, pin_memory):
    logging.info('Loading DataLoaders...')
    # Training DataLoader
    # set arguments based on GPU or CPU destination
    if device.type == 'cuda':
        train_args = dict(shuffle=True,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=pin_memory)
    else:
        train_args = dict(shuffle=True,
                          batch_size=batch_size)

    train_loader = DataLoader(train_data, **train_args)

    # Validation DataLoader
    # set arguments based on GPU or CPU destination
    if device.type == 'cuda':
        val_args = dict(shuffle=False,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory)
    else:
        val_args = dict(shuffle=False,
                        batch_size=batch_size)

    val_loader = DataLoader(val_data, **val_args)

    # Test DataLoader
    test_args = val_args  # same as validation dataset
    test_loader = DataLoader(test_data, **test_args)

    logging.info('DataLoaders loaded.')
    return train_loader, val_loader, test_loader


def load(data_path, train_data_file, train_label_file, val_data_file, val_label_file, test_data_file, train_val_dataset,
         test_dataset, device, batch_size, num_workers, pin_memory, **kwargs):
    # load Datasets
    train_data, val_data, test_data = load_dataset(data_path, train_data_file, train_label_file, val_data_file,
                                                   val_label_file,
                                                   test_data_file, train_val_dataset,
                                                   test_dataset, **kwargs)

    # load DataLoaders
    train_loader, val_loader, test_loader = load_dataloaders(train_data, val_data, test_data, device, batch_size,
                                                             num_workers, pin_memory)

    return train_loader, val_loader, test_loader
