"""
All things related to model checkpoints.
"""

import os
import logging

import torch

from octopus.utilities import utilities


class CheckpointHandler:
    def __init__(self, checkpoint_dir, delete_existing_checkpoints, run_name):
        self.checkpoint_dir = checkpoint_dir
        self.delete_existing_checkpoints = delete_existing_checkpoints
        self.run_name = run_name

    def setup(self):
        logging.info('Setting up checkpoint handler...')

        logging.info('Preparing checkpoint directory...')
        if self.delete_existing_checkpoints:
            utilities.delete_directory(self.checkpoint_dir)

        utilities.create_directory(self.checkpoint_dir)


    def save(self, model, optimizer, scheduler, next_epoch, stats):
        # build filename
        filename = os.path.join(self.checkpoint_dir, f'{self.run_name}.checkpoint.{next_epoch - 1}.pt')
        logging.info(f'Saving checkpoint to {filename}...')

        # build state dictionary
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'next_epoch': next_epoch,
            'stats': stats
        }

        torch.save(checkpoint, filename)

    def load(self, filename, device, model, optimizer, scheduler):
        logging.info(f'Loading checkpoint from {filename}...')
        checkpoint = torch.load(filename, map_location=device)

        # reload saved states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint
