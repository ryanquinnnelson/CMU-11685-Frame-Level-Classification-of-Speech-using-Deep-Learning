"""
Runs octopus.
"""

import configparser
import sys

from octopus.octopus import Octopus


def main():
    # parse config file
    config_path = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_path)

    # run octopus
    octopus = Octopus(config)
    octopus.setup_environment()
    octopus.download_data()
    octopus.run_pipeline()
    octopus.cleanup()


if __name__ == "__main__":
    # execute only if run as a script
    main()
