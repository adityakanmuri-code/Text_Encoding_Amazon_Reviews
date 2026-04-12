from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler 

class model_trainer():
    def __split_data(self,X,y):
        try:
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=43)
            return X_train,X_test,y_train,y_test
        except Exception as e:
            raise Exception(e)
    
    def __reduce_imbalances(self,X_train,y_train):
        try:
            ros = RandomOverSampler()
            X_train_resampled,y_train_resampled = ros.fit_resample(X_train,y_train)
            return X_train_resampled,y_train_resampled
        except Exception as e:
            raise Exception(e)
    
    def __encode_text_BOW(self,X,y):
        try:
            X_train,X_test,y_train,y_test = self.__split_data(X,y)
            bow = CountVectorizer(max_features=2500,ngram_range=(2,3))
            X_train_BOW = bow.fit_transform(X_train)
            X_test_BOW = bow.transform(X_test)
            X_train_BOW_resampled,y_train_resampled = self.__reduce_imbalances(X_train_BOW,y_train)
            return X_train_BOW_resampled,X_test_BOW,y_train_resampled,y_test
        except Exception as e:
            raise Exception(e)
    
    def __encode_text_BOW(self,X,y):
        try:
            X_train,X_test,y_train,y_test = self.__split_data(X,y)
            tfidf = TfidfVectorizer(max_features=2500,ngram_range=(2,3))
            X_train_tfidf = tfidf.fit_transform(X_train)
            X_test_tfidf = tfidf.transform(X_test)
            X_train_tfidf_resampled,y_train_resampled = self.__reduce_imbalances(X_train_tfidf,y_train)
            return X_train_tfidf_resampled,X_test_tfidf,y_train_resampled,y_test
        except Exception as e:
            raise Exception(e)
        
    def train_model(self,X,y):
        try:
            pass
        except Exception as e:
            raise Exception(e)
