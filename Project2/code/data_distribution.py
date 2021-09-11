from sklearn.datasets import fetch_20newsgroups
import pandas as pds
import numpy as np
import matplotlib.pyplot as plt


news_train = fetch_20newsgroups(subset='train', remove=('headers' 'footers', 'quotes'))
news_test = fetch_20newsgroups(subset='test', remove=('headers' 'footers', 'quotes'))
df_train = pds.DataFrame({'content': news_train.data, 'label': news_train.target})
df_test = pds.DataFrame({'content': news_test.data, 'label': news_test.target})
y_train = df_train['label'].to_numpy()
y_test = df_test['label'].to_numpy()
train_dist = np.bincount(y_train)
test_dist = np.bincount(y_test)
print(np.sum(train_dist))
print(np.sum(test_dist))
train_dist = 100.0*train_dist/np.sum(train_dist)
labels = news_train.target_names

IMDB_train = pds.read_csv('train.csv')
IMDB_test = pds.read_csv('test.csv')
print(IMDB_train.shape[0])
print(IMDB_test.shape[0])


fig1, ax1 = plt.subplots()
plt.title('Class Distribution of 20 News')
patches, texts, autotexts = ax1.pie(train_dist, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
[_.set_fontsize(8) for _ in texts]
plt.show()
