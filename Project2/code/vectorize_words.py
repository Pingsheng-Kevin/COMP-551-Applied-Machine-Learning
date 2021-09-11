from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pds


class Vectorizer():

    def __init__(self, option='binary', sublinear_tf=False):
        if option not in {'binary', 'tfidf', 'count'}:
            raise ValueError('option must be one of {"binary", "tfidf", "count"}')
        if option is 'binary':
            self.vectorizer = CountVectorizer(binary=True)
        elif option is 'tfidf':
            self.vectorizer = TfidfVectorizer(sublinear_tf=sublinear_tf)
        elif option is 'count':
            self.vectorizer = CountVectorizer()

    def vectorize_csv(self, csv_path,
                       ngram_range=(1, 1), remove_stop_words=False, vocabulary=None):
        self.vectorizer.ngram_range = ngram_range
        if remove_stop_words:
            self.vectorizer.stop_words = 'english'
        if vocabulary is not None:
            self.vectorizer.vocabulary = vocabulary

        with open(csv_path, 'r') as csv:
            df = pds.read_csv(csv)
            X = self.vectorizer.fit_transform(df['content']).toarray()
            y = df['label'].to_numpy()
        return X, y, self.vectorizer.get_feature_names()

    def vectorize_df_text(self, df_text,
                       ngram_range=(1, 1), remove_stop_words=False, vocabulary=None):
        self.vectorizer.ngram_range = ngram_range
        if remove_stop_words:
            self.vectorizer.stop_words = 'english'
        if vocabulary is not None:
            self.vectorizer.vocabulary = vocabulary

        X = self.vectorizer.fit_transform(df_text['content']).toarray()
        y = df_text['label'].to_numpy()
        # return vectorized text and label, and a bag of words
        return X, y, self.vectorizer.get_feature_names()
