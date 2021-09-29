import logging
import os
import shutil


def create_directory(path):
    if os.path.isdir(path):
        logging.info(f'Directory already exists:{path}.')
    else:
        os.mkdir(path)
        logging.info(f'Created directory:{path}.')


def delete_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        logging.info(f'Deleted directory:{path}.')
    else:
        logging.info(f'Directory does not exist:{path}.')