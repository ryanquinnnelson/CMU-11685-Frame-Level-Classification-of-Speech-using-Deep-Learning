"""
Common utilities.
"""
__author__ = 'ryanquinnnelson'

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


def delete_file(path):
    if os.path.isfile(path):
        os.remove(path)
        logging.info(f'Deleted file:{path}')
    else:
        logging.info(f'File does not exist:{path}')
