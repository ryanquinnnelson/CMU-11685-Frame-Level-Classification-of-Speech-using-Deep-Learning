import pandas as pd
import sys


def format(results_file):
    # read in file
    results_df = pd.read_csv(results_file, header=None)

    # change column name
    results_df = results_df.reset_index()
    results_df = results_df.rename(columns={0: "label", 'index': 'id'})
    print(results_df.shape)

    # save file
    results_df.to_csv('formatted.' + results_file, index=False)


def main():
    format_results(sys.argv[1])


if __name__ == "__main__":
    # execute only if run as a script
    main()
