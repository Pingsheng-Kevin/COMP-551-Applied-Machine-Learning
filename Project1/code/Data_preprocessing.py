import numpy as np
import pandas as pds
import argparse
import warnings


def preprocess_hepatitis_data(filepath):
    # load and clean the hepatitis data
    df = pds.read_csv(filepath)
    df_clean = df[~df.eq('?').any(1)]
    numpy_data = df_clean.to_numpy(dtype=float)
    numpy_labels = numpy_data[:, -1].astype(int)
    labels_dist = np.bincount(numpy_labels)
    labels_sum = np.sum(labels_dist)
    print('The number of Class 1 hepatitis (LIVE) in the cleaned dataset is: ', labels_dist[1],
          f'({100 * labels_dist[1] / labels_sum:.2f}%)')
    print('The number of Class 2 hepatitis (DIE) in the cleaned dataset is: ', labels_dist[2],
          f'({100 * labels_dist[2] / labels_sum:.2f}%)')

    np_sex = np.bincount(df_clean['SEX'].to_numpy(dtype=int))
    sex_total = np.sum(np_sex)
    print(f'Number of males: {np_sex[1]} ({100*np_sex[1]/sex_total}%)')
    print(f'Number of females: {np_sex[2]} ({100*np_sex[2]/sex_total}%)')
    return df_clean


def preprocess_cancer_data(filepath):
    # load and clean the cancer data
    # all features in data are in range of 1-10, integer value.
    df = pds.read_csv(filepath)
    df_clean = df[~df.eq('?').any(1)]
    df_clean = df_clean.drop('id', axis='columns')
    df_clean['Class'] = df_clean['Class'].__floordiv__(2)
    numpy_data = df_clean.to_numpy(dtype=int)
    numpy_labels = numpy_data[:, -1]
    # print(numpy_labels)
    labels_dist = np.bincount(numpy_labels)
    labels_sum = np.sum(labels_dist)
    print('The number of Class 2 tumors in the cleaned dataset is: ', labels_dist[1],
          f'({100*labels_dist[1]/labels_sum:.2f}%)')
    print('The number of Class 4 tumors in the cleaned dataset is: ', labels_dist[2],
          f'({100*labels_dist[2]/labels_sum:.2f}%)')
    return df_clean


def main():
    warnings.filterwarnings(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("-c", "--cancer", action='store_true', help="this flag means the input is the cancer file")
    parser.add_argument("-H", "--hepatitis", action='store_true')
    args = parser.parse_args()
    if args.cancer:
        df = preprocess_cancer_data(args.data_path)

    elif args.hepatitis:
        df = preprocess_hepatitis_data(args.data_path)

    else:
        print('This file is unknown, therefore unable to process.')
        quit(0)


if __name__ == '__main__':
    main()
