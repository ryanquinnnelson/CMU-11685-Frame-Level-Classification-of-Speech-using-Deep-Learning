"""
All things related to data reading and writing.
"""

import os
import logging
from datetime import datetime

import pandas as pd
from torch.utils.data import DataLoader

from octopus.utilities import utilities


class DataHandler:

    def __init__(self,
                 run_name,
                 data_dir,
                 output_dir,
                 train_data_file,
                 train_label_file,
                 val_data_file,
                 val_label_file,
                 test_data_file,
                 test_label_file,
                 batch_size,
                 num_workers,
                 pin_memory,
                 dataset_kwargs):

        self.run_name = run_name
        self.data_dir = data_dir
        self.output_dir = output_dir

        # fully-qualified file names
        self.train_data_file = os.path.join(data_dir, train_data_file)
        self.train_label_file = os.path.join(data_dir, train_label_file)
        self.val_data_file = os.path.join(data_dir, val_data_file)
        self.val_label_file = os.path.join(data_dir, val_label_file)
        self.test_data_file = os.path.join(data_dir, test_data_file)
        if test_label_file:
            self.test_label_file = os.path.join(data_dir, test_label_file)
        else:
            self.test_label_file = None

        # parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_kwargs = dataset_kwargs

    def setup(self):
        logging.info('Setting up data handler...')

        logging.info('Preparing output directory...')
        utilities.create_directory(self.output_dir)

    def train_dataset(self, dataset_class):

        ds = dataset_class(self.train_data_file, self.train_label_file, **self.dataset_kwargs)
        logging.info(f'Loaded {ds.length} records as training data.')
        return ds

    def val_dataset(self, dataset_class):

        ds = dataset_class(self.val_data_file, self.val_label_file, **self.dataset_kwargs)
        logging.info(f'Loaded {ds.length} records as validation data.')
        return ds

    def test_dataset(self, dataset_class):

        ds = dataset_class(self.test_data_file, self.test_label_file, **self.dataset_kwargs)
        logging.info(f'Loaded {ds.length} records as test data.')
        return ds

    def train_dataloader(self, dataset, device):
        # set arguments based on GPU or CPU destination
        if device.type == 'cuda':
            train_args = dict(shuffle=True,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              pin_memory=self.pin_memory)
        else:
            train_args = dict(shuffle=True,
                              batch_size=self.batch_size)

        dl = DataLoader(dataset, **train_args)
        return dl

    def val_dataloader(self, dataset, device):
        # set arguments based on GPU or CPU destination
        if device.type == 'cuda':
            val_args = dict(shuffle=False,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)
        else:
            val_args = dict(shuffle=False,
                            batch_size=self.batch_size)

        dl = DataLoader(dataset, **val_args)
        return dl

    def test_dataloader(self, dataset, device):
        return self.val_dataloader(dataset, device)  # same configs as validation set

    def load(self, train_dataset_class, val_dataset_class, test_dataset_class, devicehandler):

        logging.info('Loading data...')

        # Datasets
        train_dataset = self.train_dataset(train_dataset_class)
        val_dataset = self.val_dataset(val_dataset_class)
        test_dataset = self.test_dataset(test_dataset_class)

        # DataLoaders
        device = devicehandler.get_device()
        train_dl = self.train_dataloader(train_dataset, device)
        val_dl = self.val_dataloader(val_dataset, device)
        test_dl = self.test_dataloader(test_dataset, device)

        return train_dl, val_dl, test_dl

    def save(self, out, epoch):

        # generate filename
        filename = f'{self.run_name}.epoch{epoch}.{datetime.now().strftime("%Y%m%d.%H.%M.%S")}.output.csv'
        path = os.path.join(self.output_dir, filename)

        logging.info(f'Saving test output to {path}...')

        # save output
        df = pd.DataFrame(data=out)
        df.to_csv(path, header=False)
