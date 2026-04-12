from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler


class model_trainer():

    def __split_data(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def __reduce_imbalances(self, X_train, y_train):
        ros = RandomOverSampler(random_state=42)
        return ros.fit_resample(X_train, y_train)

    def __encode_text_BOW(self, X, y):
        X_train, X_test, y_train, y_test = self.__split_data(X, y)

        bow = CountVectorizer(max_features=2500, ngram_range=(1, 3))
        X_train_bow = bow.fit_transform(X_train)
        X_test_bow = bow.transform(X_test)

        X_train_resampled, y_train_resampled = self.__reduce_imbalances(X_train_bow, y_train)

        return X_train_resampled, X_test_bow, y_train_resampled, y_test, bow

    def __encode_text_TFIDF(self, X, y):
        X_train, X_test, y_train, y_test = self.__split_data(X, y)

        tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 3))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        X_train_resampled, y_train_resampled = self.__reduce_imbalances(X_train_tfidf, y_train)

        return X_train_resampled, X_test_tfidf, y_train_resampled, y_test, tfidf

    def train_model(self, X, y, encode_tech: str = 'TFIDF'):
        if encode_tech == 'BOW':
            X_train, X_test, y_train, y_test, vectorizer = self.__encode_text_BOW(X, y)
        elif encode_tech == 'TFIDF':
            X_train, X_test, y_train, y_test, vectorizer = self.__encode_text_TFIDF(X, y)
        else:
            raise ValueError("Invalid encoding type. Use 'BOW' or 'TFIDF'")

        model = MultinomialNB()
        model.fit(X_train, y_train)

        return model, X_test, y_test, vectorizer


class model_eval():
    def model_eval(self, model, X_test, y_test, encoding: str):
        y_pred = model.predict(X_test)

        return {
            'Encoding': encoding,
            'Model_Name': model.__class__.__name__,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        }
