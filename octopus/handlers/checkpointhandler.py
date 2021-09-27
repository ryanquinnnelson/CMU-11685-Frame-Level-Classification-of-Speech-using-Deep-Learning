"""
All things related to model checkpoints.
"""

import os
import logging
import shutil

import torch


class CheckpointHandler:
    def __init__(self, checkpoint_dir, delete_existing_checkpoints, run_name):
        self.checkpoint_dir = checkpoint_dir
        self.delete_existing_checkpoints = delete_existing_checkpoints
        self.run_name = run_name

    def setup(self):
        logging.info('Setting up checkpoint directory...')
        if self.delete_existing_checkpoints:
            _delete_directory(self.checkpoint_dir)

        _create_directory(self.checkpoint_dir)
        logging.info('Checkpoint directory set up.')

    def save(self, model, optimizer, scheduler, next_epoch, stats):
        logging.info('Saving checkpoint...')

        # build state dictionary
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'next_epoch': next_epoch,
            'stats': stats
        }

        # save file
        filename = os.path.join(self.checkpoint_dir, f'{self.run_name}.checkpoint.{next_epoch - 1}.pt')
        torch.save(checkpoint, filename)
        logging.info(f'Saved checkpoint to {filename}.')

    def load(self, filename, device, model, optimizer, scheduler):
        logging.info(f'Loading checkpoint from {filename}...')
        checkpoint = torch.load(filename, map_location=device)

        # reload saved states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logging.info('Checkpoint loaded.')

        return checkpoint


def _create_directory(path):
    if os.path.isdir(path):
        logging.info(f'Directory already exists:{path}.')
    else:
        os.mkdir(path)
        logging.info(f'Created directory:{path}.')


def _delete_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        logging.info(f'Deleted directory:{path}.')
