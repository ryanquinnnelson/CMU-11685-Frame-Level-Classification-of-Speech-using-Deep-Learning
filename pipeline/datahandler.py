"""
All things related to data reading and writing.
"""

import os
import logging

from torch.utils.data import DataLoader


class DataDealer:

    def __init__(self,
                 data_dir,
                 train_data_file,
                 train_label_file,
                 val_data_file,
                 val_label_file,
                 test_data_file,
                 test_label_file=None,
                 batch_size=16,
                 num_workers=0,
                 pin_memory=False):

        self.data_dir = data_dir
        self.train_data_file = os.path.join(data_dir, train_data_file)
        self.train_label_file = os.path.join(data_dir, train_label_file)
        self.val_data_file = os.path.join(data_dir, val_data_file)
        self.val_label_file = os.path.join(data_dir, val_label_file)
        self.test_data_file = os.path.join(data_dir, test_data_file)
        if test_label_file:
            self.test_label_file = os.path.join(data_dir, test_label_file)
        else:
            self.test_label_file = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataset(self, dataset_class, **kwargs):

        ds = dataset_class(self.train_data_file, self.train_label_file, **kwargs)
        logging.info(f'Loaded {ds.length} records as training data.')
        return ds

    def val_dataset(self, dataset_class, **kwargs):

        ds = dataset_class(self.val_data_file, self.val_label_file, **kwargs)
        logging.info(f'Loaded {ds.length} records as validation data.')
        return ds

    def test_dataset(self, dataset_class, **kwargs):

        ds = dataset_class(self.test_data_file, self.test_label_file, **kwargs)
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

    def load(self, train_dataset_class, val_dataset_class, test_dataset_class, device, **kwargs):

        # Datasets
        train_dataset = self.train_dataset(train_dataset_class, **kwargs)
        val_dataset = self.val_dataset(val_dataset_class, **kwargs)
        test_dataset = self.test_dataset(test_dataset_class, **kwargs)

        # DataLoaders
        train_dl = self.train_dataloader(train_dataset, device)
        val_dl = self.val_dataloader(val_dataset, device)
        test_dl = self.test_dataloader(test_dataset, device)

        return train_dl, val_dl, test_dl
