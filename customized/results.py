import pandas as pd
import sys


def format(results_file):
    # read in file
    results_df = pd.read_csv(results_file, header=None)
    print(results_df.columns)

    # change column name
    # results_df = results_df.reset_index()
    results_df = results_df.rename(columns={1: "label", 0: 'id'})
    # results_df = results_df.rename(columns={0: "label", 'index': 'id'})
    print(results_df)
    # print(results_df.shape)
    #
    # # save file
    results_df.to_csv(results_file + 'formatted.csv', index=False)


def main():
    format(sys.argv[1])


if __name__ == "__main__":
    # execute only if run as a script
    main()
