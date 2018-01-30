
import sys
import os
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def write_data(rawdir, dataset, targetdir, label_col, file_col)
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    for _, row in dataset.iterrows():
        file = row[file_col]
        label = row[label_col]
        if not os.path.exists(labeldir):
            os.makedirs(labeldir)
        srcfile = rawdir + '/' + file
        tgtfile = labeldir + '/' + file
        shutil.copyfile(srcfile, tgtfile)


def split_train_test_data(datadir, label):
    rawdir = datadir + "/raw"
    data = rawdir + "/data.csv"
    datafile = pd.read_csv(data, spe='.', header=0)
    train, test = train_test_split(datafile, test_size=0.2, shuffle=True)
    file_col = 'FileName'
    # setup train data and directory
    write_data(rawdir, train, datadir + "/train", label, file_col)
    # setup test data and directory
    write_data(rawdir, test, datadir + "/test", label, file_col)
    print("All done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("datadir",
            help="The working data directory.")
    parser.add_argument("--label", default="Type",
            help="label from data to be used.")
    args = parser.parse_args()
    return split_train_test_data(args.datadir, args.label)

if __name__ == '__main__':
    sys.exit(main())
