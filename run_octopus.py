"""
Runs octopus.
"""

import configparser
import sys

from octopus import octopus


def main():
    # parse config file
    config_path = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_path)

    # run octopus
    oct = octopus.Octopus(config)
    oct.setup_environment()
    oct.download_data()
    oct.run_pipeline()


if __name__ == "__main__":
    # execute only if run as a script
    main()
