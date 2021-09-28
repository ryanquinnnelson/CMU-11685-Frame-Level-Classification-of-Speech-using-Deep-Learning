"""
All things related to kaggle.
"""

import logging
import os
import subprocess
import json
import glob
import zipfile


class KaggleConnector:

    def __init__(self, kaggle_dir, content_dir, token_file, competition, delete_zipfiles):
        self.kaggle_dir = kaggle_dir
        self.content_dir = content_dir
        self.token_file = token_file
        self.competition = competition
        self.competition_dir = os.path.join(content_dir, 'competitions', competition)
        self.delete_zipfiles = delete_zipfiles

    def setup(self):
        logging.info('Setting up kaggle...')

        # download and install library
        _install()

        # create directories
        _mkdirs(self.kaggle_dir, self.content_dir)

        # setup for kaggle api token
        token = _read_kaggle_token(self.token_file)
        token_dest = _write_kaggle_token(token, self.kaggle_dir)
        _secure_kaggle_token(token_dest)

        # configure kaggle to use content directory
        _configure_content_dir(self.content_dir)

        logging.info('kaggle is set up.')

    def download(self):
        if not os.path.isdir(self.competition_dir):
            logging.info(f'Downloading kaggle competition:{self.competition}...')
            process = subprocess.Popen(['kaggle', 'competitions', 'download', '-c', self.competition],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            logging.info(stdout.decode("utf-8"))
            logging.info('Competition files downloaded.')
        else:
            logging.info('Competition files are already downloaded.')

    def unzip(self):
        if not os.path.isdir(self.competition_dir):
            logging.info('Unzipping competition files...')
            # get filenames
            zipfiles = glob.glob(self.competition_dir + '/*.zip')

            # verify files exist
            if len(zipfiles) == 0:
                raise ValueError("No files were found.")

            # unzip each file
            for f in zipfiles:
                with zipfile.ZipFile(f, 'r') as zip_ref:
                    zip_ref.extractall(self.competition_dir)

            # clean up original zipfile
            if self.delete_zipfiles:
                for f in zipfiles:
                    os.remove(f)

            logging.info('Competition files unzipped.')
        else:
            logging.info('Competition files are already unzipped.')

    def download_and_unzip(self):
        self.download()
        self.unzip()


def _configure_content_dir(content_dir):
    process = subprocess.Popen(['kaggle', 'config', 'set', '-n', 'path', '-v', content_dir],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))


def _mkdirs(kaggle_dir, content_dir):
    # kaggle directory
    if not os.path.isdir(kaggle_dir):
        os.mkdir(kaggle_dir)

    # kaggle content directory
    if not os.path.isdir(content_dir):
        os.mkdir(content_dir)


def _read_kaggle_token(token_file):
    with open(token_file) as token_source:
        token = json.load(token_source)
        return token


def _write_kaggle_token(token, kaggle_dir):
    token_dest = os.path.join(kaggle_dir, 'kaggle.json')
    with open(token_dest, 'w') as file:
        json.dump(token, file)

    return token_dest


def _secure_kaggle_token(token_dest):
    process = subprocess.Popen(['chmod', '600', token_dest],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))


def _install():
    logging.info('Installing kaggle...')
    process = subprocess.Popen(['pip', 'install', 'kaggle'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))
    logging.info('kaggle installed.')
