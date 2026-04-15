from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


class model_trainer():

    def __split_data(self, X, y):
        X_train,X_temp,y_train,y_temp = train_test_split(X,y,test_size=0.4,stratify=y)
        X_val,X_test,y_val,y_test = train_test_split(X_temp,y_temp,test_size=0.2,stratify=y_temp)
        return X_train,X_val,X_test,y_train,y_val,y_test

    def __reduce_imbalances(self, X_train, y_train):
        ros = RandomOverSampler(random_state=42)
        return ros.fit_resample(X_train, y_train)

    def __encode_text_BOW(self, X, y):
        X_train,X_val,X_test,y_train,y_val,y_test = self.__split_data(X, y)

        bow = CountVectorizer(max_features=2500, ngram_range=(1, 3))
        X_train_bow = bow.fit_transform(X_train)
        X_test_bow = bow.transform(X_test)
        X_val_bow = bow.transform(X_val)

        X_train_resampled, y_train_resampled = self.__reduce_imbalances(X_train_bow, y_train)

        return X_train_resampled, X_test_bow, y_train_resampled, y_test, bow

    def __encode_text_TFIDF(self, X, y):
        X_train,X_val,X_test,y_train,y_val,y_test = self.__split_data(X, y)

        tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 3))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        X_val_tfidf = tfidf.transform(X_val)

        X_train_resampled, y_train_resampled = self.__reduce_imbalances(X_train_tfidf, y_train)

        return X_train_resampled, X_test_tfidf, y_train_resampled, y_test, tfidf

    def calculate_cv_score(model,X,y):
        return cross_val_score(model,X,y,cv=5,scoring='f1_weighted').mean()

    def train_model(self, X, y, encode_tech: str = 'TFIDF'):
        if encode_tech == 'BOW':
            X_train,X_val,X_test,y_train,y_val,y_test,vectorizer = self.__encode_text_BOW(X, y)
        elif encode_tech == 'TFIDF':
            X_train,X_val,X_test,y_train,y_val,y_test,vectorizer = self.__encode_text_TFIDF(X, y)
        else:
            raise ValueError("Invalid encoding type. Use 'BOW' or 'TFIDF'")

        models = {
            'Naive_Bayes' : MultinomialNB(),
            'Logistic_Regression' : LogisticRegression(),
            'Linear_SVC' : LinearSVC()
        }
        best_score = -1
        best_model = None

        for name,model in models.items():
            model.fit(X_train,y_train)
            scores = self.model_eval(model,X_test,y_test)
            cv_score = self.calculate_cv_score(model,X_test,y_test)
            #Fetching the best model 
            if cv_score >= best_score:
                if scores['F1_Score'] > best_score:
                    best_model = model
                    best_model_name = model.__class__.__name__
                    best_score = scores['F1_score']
    
    def model_eval(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        return {
            'Model_Name': model.__class__.__name__,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred,average='weighted'),
            'Confusion Matrix' : confusion_matrix(y_test,y_pred),
            'Classification Report' : classification_report(y_test,y_pred)
        }
