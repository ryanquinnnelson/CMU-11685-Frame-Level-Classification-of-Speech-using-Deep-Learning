import configparser
import sys

from octopus import octopus


def main():
    # parse config file
    config_path = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_path)

    # run octopus
    oc = octopus.Octopus(config)
    oc.setup_environment()
    oc.run_pipeline()


if __name__ == "__main__":
    # execute only if run as a script
    main()
