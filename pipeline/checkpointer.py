import os
import logging
import shutil
import torch


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


def setup_checkpoint_directory(checkpoint_dir, delete_existing_checkpoints):
    logging.info('Setting up checkpoint directory...')
    if delete_existing_checkpoints:
        _delete_directory(checkpoint_dir)

    _create_directory(checkpoint_dir)
    logging.info('Checkpoint directory set up.')


def save_checkpoint(model, optimizer, scheduler, next_epoch, max_val_acc, checkpoint_dir, name):
    logging.info('Saving checkpoint...')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'next_epoch': next_epoch,
        'max_val_acc': max_val_acc
    }

    filename = os.path.join(checkpoint_dir, f'{name}.checkpoint.{next_epoch - 1}.pt')
    torch.save(checkpoint, filename)
    logging.info(f'Saved checkpoint to {filename}.')


def load_checkpoint(model, optimizer, scheduler, filename, device):
    logging.info(f'Loading checkpoint from {filename}...')
    checkpoint = torch.load(filename, map_location=device)

    # reload saved states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    logging.info('Checkpoint loaded.')

    return checkpoint
