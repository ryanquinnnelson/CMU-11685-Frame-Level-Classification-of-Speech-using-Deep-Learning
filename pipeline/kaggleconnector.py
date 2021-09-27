import logging
import os
import subprocess
import json
import glob
import zipfile


def setup(kaggle_dir, content_dir, token_file):
    logging.info('Setting up kaggle...')

    # install
    process = subprocess.Popen(['pip', 'install', 'kaggle'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))

    # create kaggle directory
    if not os.path.isdir(kaggle_dir):
        os.mkdir(kaggle_dir)

    # read in kaggle token
    with open(token_file) as token_source:
        token = json.load(token_source)

    # write kaggle token to kaggle directory
    token_dest = os.path.join(kaggle_dir, 'kaggle.json')
    with open(token_dest, 'w') as file:
        json.dump(token, file)

    # secure the kaggle token file
    process = subprocess.Popen(['chmod', '600', token_dest],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))

    # create kaggle content directory
    if not os.path.isdir(content_dir):
        os.mkdir(content_dir)

    # configure kaggle to use content directory
    process = subprocess.Popen(['kaggle', 'config', 'set', '-n', 'path', '-v', content_dir],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))

    logging.info('kaggle is set up.')


def download(competition):
    logging.info(f'Downloading kaggle competition:{competition}...')
    process = subprocess.Popen(['kaggle', 'competitions', 'download', '-c', competition],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    logging.info(stdout.decode("utf-8"))
    logging.info('Competition files downloaded.')


def unzip(path):
    logging.info('Unzipping competition files...')
    # get filenames
    zipfiles = glob.glob(path + '/*.zip')

    # unzip each file
    for f in zipfiles:
        with zipfile.ZipFile(f, 'r') as zip_ref:
            zip_ref.extractall(path)

    # delete original zipfile
    for f in zipfiles:
        os.remove(f)

    logging.info('Competition files unzipped.')


def get_competition_path(content_dir, competition):
    competition_path = os.path.join(content_dir, 'competitions', competition)
    return competition_path
