import pandas as pds
import os
from pathlib import Path

datapath = Path(os.getcwd())

train_path = datapath/'train'
test_path = datapath/'test'


def dir_reviews_to_df(path):
    files_list = list(path.iterdir())
    reviews_list = []
    for file in files_list:
        with open(file, 'r') as f:
            reviews_list.append(f.read())
    df = pds.DataFrame({'content': reviews_list})
    return df


def label_and_load_data(dir_path):
    # train or test
    pos_path = dir_path/'pos'
    neg_path = dir_path/'neg'
    pos_reviews = dir_reviews_to_df(pos_path)
    neg_reviews = dir_reviews_to_df(neg_path)
    pos_reviews['label'] = 1
    neg_reviews['label'] = 0
    concatenated = pds.concat([pos_reviews, neg_reviews])
    concatenated.reset_index(drop=True, inplace=True)
    return concatenated

train_df = label_and_load_data(train_path)
test_df = label_and_load_data(test_path)
# sample a smaller data set
"""
dfg_train = train_df.groupby(['label'])
dfg_test = test_df.groupby(['label'])
df_train_sample = pds.DataFrame()
df_test_sample = pds.DataFrame()
for k, v in dfg_train:
    v_sample = v.head(250)
    df_train_sample = pds.concat([df_train_sample, v_sample])
for k, v in dfg_test:
    v_sample = v.head(250)
    df_test_sample = pds.concat([df_test_sample, v_sample])
df_train_sample.reset_index(drop=True, inplace=True)
df_test_sample.reset_index(drop=True, inplace=True)
df_train_sample.to_csv('IMDB_train_sample.csv', index=False)
df_test_sample.to_csv('IMDB_test_sample.csv', index=False)
"""
train_df.to_csv('train.csv', sep=',', index=False)
test_df.to_csv('test.csv', sep=',', index=False)

