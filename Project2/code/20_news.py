from sklearn.datasets import fetch_20newsgroups
from MultinomialNB import MultinomialNB as MNB
from BernoulliNB import BernoulliNB as BNB
from GaussianNB import GaussianNB as GNB
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import time
from sklearn.model_selection import GridSearchCV
from vectorize_words import Vectorizer
import pandas as pds
import nltk
nltk.download('wordnet')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

news_train = fetch_20newsgroups(subset='train', remove=('headers' 'footers', 'quotes'))
news_test = fetch_20newsgroups(subset='test', remove=('headers' 'footers', 'quotes'))
# lemmatize and clean the text
signs = list("?:!,.;@*-=\n\r\t\"()")
for sign in signs:
    news_train.data = [w.replace(sign, ' ') for w in news_train.data]
    news_test.data = [w.replace(sign, ' ') for w in news_test.data]

tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_content(content):
    return [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(content)]


df_train = pds.DataFrame({'content': news_train.data, 'label': news_train.target})
df_test = pds.DataFrame({'content': news_test.data, 'label': news_test.target})

df_train['content'] = df_train.content.apply(lemmatize_content)
df_test['content'] = df_test.content.apply(lemmatize_content)
df_train['content'] = df_train['content'].apply(' '.join)
df_test['content'] = df_test['content'].apply(' '.join)
# print(df_train['content'])


news_vectorizer = Vectorizer(option='binary')
X_train, y_train, bag_of_words = news_vectorizer.vectorize_df_text(df_train, ngram_range=(1, 2))
# print(len(bag_of_words))
X_test, y_test, words = news_vectorizer.vectorize_df_text(df_test, vocabulary=bag_of_words)


def fit_and_score(classifier, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    classifier.fit(X_tr, y_tr)
    t1 = time.time()
    print(f'fit time: {t1 - t0}s')
    score = classifier.score(X_te, y_te)
    print(f'score is: {score*100}')
    t2 = time.time()
    print(f'time used for scoring: {t2 - t1}s')
    return score*100, t1-t0, t2-t1


def report_results(results, top=10):
    for i in range(1, top+1):
        models = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in models:
            print(f"Model Rank: {i}")
            print(f"Mean validation score: {results['mean_test_score'][candidate]:.5f} "
                  f"(std: {results['std_test_score'][candidate]:.5f})")
            print(f"Parameters: {results['params'][candidate]}")
            print("")


# tuning Hyperparameters
def tuning(X, y, C_range):
    # LR
    para_dict = {'penalty': ['l2'],
                 'C': [i for i in C_range]
                 }
    classifier = LogisticRegression(solver='liblinear')
    searcher = GridSearchCV(classifier, param_grid=para_dict, cv=5)
    t0 = time.time()
    searcher.fit(X, y)
    t1 = time.time()
    print(f'Grid Search takes {t1 - t0}s to tune hyperparameters of Logistic Regression Classifier.')
    report_results(searcher.cv_results_)

    # SVM
    para_dict = {'penalty': ['l2'],
                 'C': np.logspace(-2, 5, 8)}
    classifier = LinearSVC()
    searcher = GridSearchCV(classifier, param_grid=para_dict, cv=5)
    t0 = time.time()
    searcher.fit(X, y)
    t1 = time.time()
    print(f'Grid Search takes{t1 - t0}s to tune hyperparameters of SVM Classifier.')
    report_results(searcher.cv_results_, top=8)


tuning(X_train, y_train, C_range=range(1,6))
fit_time_list = []
score_time_list = []
score_list = []
classifier = MNB()
print('fit and score the MultinominalNB model with best hyper parameter using binary occurrence-indexed data.')
score, fit_time, score_time = fit_and_score(classifier, X_train, y_train, X_test, y_test)
score_list.append(score)
score_time_list.append(score_time/10.0)
fit_time_list.append(fit_time/10.0)
# raise Exception()
classifier = GNB()
print('fit and score the GaussianNB model with best hyper parameter using binary occurrence-indexed data.')
score, fit_time, score_time = fit_and_score(classifier, X_train, y_train, X_test, y_test)
score_list.append(score)
score_time_list.append(score_time/10.0)
fit_time_list.append(fit_time/10.0)

classifier = LinearSVC(C=1.0)
print('fit and score the LinearSVC model with best hyper parameter using binary occurrence indexed data.')
score, fit_time, score_time = fit_and_score(classifier, X_train, y_train, X_test, y_test)
score_list.append(score)
score_time_list.append(score_time/10.0)
fit_time_list.append(fit_time/10.0)

classifier = LogisticRegression(solver='liblinear', C=3)
print('fit and score the LogisticRegression model with best hyper parameter using binary occurrence indexed data.')
score, fit_time, score_time = fit_and_score(classifier, X_train, y_train, X_test, y_test)
score_list.append(score)
fit_time_list.append(fit_time/10.0)
score_time_list.append(score_time/10.0)
# plot the bars
groups = len(score_time_list)
fig, ax = plt.subplots()
index = np.arange(groups)
bar_width = 0.20
# 3 bars
plt.bar(index+bar_width, fit_time_list, bar_width, alpha=1, label='Fit Time (unit: 10s)', color='b')
plt.bar(index, score_list, bar_width, alpha=1, label='Accuracy%', color='r')
plt.bar(index+2*bar_width, score_time_list, bar_width, alpha=1, label='Score Time(unit: 10s)', color='y')
plt.xlabel('Model Name')
plt.ylabel('Score')
plt.title('Scores of each model (Binary Occurrence)')
plt.xticks(index + bar_width-0.2, ('MultinomialNB', 'GaussianNB', 'LinearSVC', 'LR'))
plt.legend()

plt.tight_layout()
plt.savefig('Scores of each model (Binary Occurrence).png', dpi=300, bbox_inches='tight')
plt.clf()


# Using TFIDF
news_vectorizer = Vectorizer(option='tfidf', sublinear_tf=True)

X_train, y_train, bag_of_words = news_vectorizer.vectorize_df_text(df_train, remove_stop_words=True,
                                                                   ngram_range=(1, 1))

X_test, y_test, words = news_vectorizer.vectorize_df_text(df_test, vocabulary=bag_of_words)


x_list = [.2*i*100 for i in range(1, 6)]
y_list = []
y_list2 = []
mnb = MNB()
lr = LogisticRegression(solver='liblinear', C=1)
for i in range(1, 6):
    np.random.seed(2021)
    percentile = .2*i
    rows = np.random.binomial(1, percentile, size=len(X_train)).astype(bool)
    X_train_sample = X_train[rows]
    y_train_sample = y_train[rows]
    print(f'Multinomial NB classifier using {percentile*100}% of the training data:')
    result, fit_time, score_time = fit_and_score(mnb, X_train_sample, y_train_sample, X_test, y_test)
    y_list.append(result)
    y_train_sample = y_train[rows]
    print(f'Logistic Regression classifier using {percentile*100}% of the training data:')
    result, fit_time, score_time = fit_and_score(lr, X_train_sample, y_train_sample, X_test, y_test)
    y_list2.append(result)

plt.plot(x_list, y_list, marker='o', color='skyblue', linewidth=2, label='Multinomial NB classifier')
plt.plot(x_list, y_list2, marker='x', color='green', linewidth=2, label='Logistic Regression classifier')
plt.xlabel('training data sample size (%)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. training data sample size (20 news) ')
plt.legend()
plt.savefig('Training data vs Accuracy.png', dpi=300, bbox_inches='tight')
plt.clf()


tuning(X_train, y_train, C_range=range(12, 22))
fit_time_list = []
score_time_list = []
score_list = []
classifier = LinearSVC(C=1.0)
print('fit and score the LinearSVC model with best hyper parameter using tfidf-indexed data.')
score, fit_time, score_time = fit_and_score(classifier, X_train, y_train, X_test, y_test)
score_list.append(score)
fit_time_list.append(fit_time/10.0)
score_time_list.append(score_time/10.0)

classifier = LogisticRegression(solver='liblinear', C=20)
print('fit and score the LogisticRegression model with best hyper parameter using tfidf-indexed data.')
score, fit_time, score_time = fit_and_score(classifier, X_train, y_train, X_test, y_test)
score_list.append(score)
fit_time_list.append(fit_time/10.0)
score_time_list.append(score_time/10.0)

classifier = GNB()
print('fit and score the GaussaianNB model with best hyper parameter using tfidf-indexed data.')
score, fit_time, score_time = fit_and_score(classifier, X_train, y_train, X_test, y_test)
score_list.append(score)
fit_time_list.append(fit_time/10.0)
score_time_list.append(score_time/10.0)

classifier = MNB()
print('fit and score the MultinomialNB model with best hyper parameter using tfidf-indexed data.')
score, fit_time, score_time = fit_and_score(classifier, X_train, y_train, X_test, y_test)
score_list.append(score)
fit_time_list.append(fit_time/10.0)
score_time_list.append(score_time/10.0)
# plot the bars
groups = len(score_time_list)
fig, ax = plt.subplots()
index = np.arange(groups)
bar_width = 0.20
plt.bar(index+bar_width, fit_time_list, bar_width, alpha=1, label='Fit Time (unit: 10s)', color='b')
plt.bar(index, score_list, bar_width, alpha=1, label='Accuracy%', color='r')
plt.bar(index+2*bar_width, score_time_list, bar_width, alpha=1, label='Score Time (unit: 10s)', color='y')
plt.xlabel('Model Name')
plt.ylabel('Score')
plt.title('Scores of each model (TF-IDF)')
plt.xticks(index + bar_width-0.2, ('LinearSVC', 'LR', 'GaussianNB', 'MultinomialNB'))
plt.legend()

plt.tight_layout()
plt.savefig('Scores of each model (TF-IDF).png', dpi=300, bbox_inches='tight')
plt.clf()

acc_list = []
acc_list2 = []
C_list = [i for i in range(1, 22, 2)]

for i in range(1, 22, 2):
    C = i
    lr = LogisticRegression(solver='liblinear', C=C)
    acc, fit_time, score_time = fit_and_score(lr, X_train, y_train, X_test, y_test)
    acc_list.append(acc)
    svm = LinearSVC(C=C)
    acc2, fit_time, score_time = fit_and_score(svm, X_train, y_train, X_test, y_test)
    acc_list2.append(acc2)


plt.plot(C_list, acc_list, marker='o', color='black', linewidth=2, label='Logistic Regression')
plt.plot(C_list, acc_list2, marker='x', color='magenta', linewidth=2, label='LinearSVC')
plt.xlabel('C Value')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. C Value')
plt.legend()
plt.savefig('C vs Accuracy(TFIDF)(20news).png', dpi=300)
plt.clf()
