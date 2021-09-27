import configparser
import sys

from pipeline import pipeline


def main():
    # parse config file
    config_path = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_path)

    # run pipeline
    pl = pipeline.Pipeline(config)
    pl.setup()
    pl.run()


if __name__ == "__main__":
    # execute only if run as a script
    main()
