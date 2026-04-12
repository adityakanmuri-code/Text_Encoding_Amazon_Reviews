'''
Text Preprocessing steps involves the following steps :
Step 1 : Text Ingestion of kaggle dataset for Amazon Reviews
Step 2 : Data Cleaning and building vocabulary for using in BOW and TFIDF
'''
import pandas as pd 
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
from tqdm import tqdm


#Step 1 : Data Ingestion
class data_ingestion():
    def ingest_data(self,file_name:str = '',file_type:str = 'csv'):
        try:
            if file_type == 'csv':
                df = pd.read_csv(file_name)
            elif file_type == 'tab':
                df = pd.read_csv(file_name,sep=' ')
            return df
        except Exception as e:
            raise Exception(e)

class text_preprocessing():
    def __init__(self,file_name:str = '',file_type:str = ''):
        self.file_name = file_name
        self.file_type = file_type
    def text_cleaning(self,unwanted_cols:list = None):
        try:
            ingest = data_ingestion()
            df = ingest.ingest_data(self.file_name,self.file_type)
            
            #Removing the unwanted columns that are not considered in the perview of the sentiment analysis
            df = self.__remove_unwanted_cols(df,unwanted_cols)
            
            #Removing nan values
            df = self.__remove_null_values(df)
            
            #Converting the text into lower case letters
            df['reviewText'] = df['reviewText'].str.lower()
            
            #Removing special characters from the review text
            df['reviewText'] = df['reviewText'].apply(lambda x : re.sub('[^a-zA-Z0-9]',' ',x))
            return df
        except Exception as e:
            raise Exception(e)
    
    def __remove_unwanted_cols(self,dframe:pd.DataFrame = None,unwanted_cols:list = None):
        try:
            return dframe.drop(columns=unwanted_cols)
        except Exception as e:
            raise Exception(e)
        
    def __remove_null_values(self,dframe:pd.DataFrame = None):
        try:
            for col in dframe.columns:
                nan_occ = dframe[col].isna().mean()
                if nan_occ >= 0.05:
                    dframe[col] = dframe[col].fillna('',axis=0)
                else:
                    dframe = dframe.dropna(subset=[col],axis=0)
            return dframe
        except Exception as e:
            raise Exception(e)
        
    def __tokenizing_corpus(self,dframe:pd.DataFrame = None):
        try:
            documents = dframe['reviewText'].tolist()
            stop_words = stopwords.words('english')
            corpus = []
            lemma = WordNetLemmatizer()
            for i in tqdm(range(len(documents))):
                documents[i] = word_tokenize(documents[i])
                documents[i] = [lemma.lemmatize(word) for word in documents[i] if word not in stop_words]
                documents[i] = ' '.join(documents[i])
                corpus.append(documents[i])
            return(corpus)
        except Exception as e:
            raise Exception(e)

    def features_setting(self,dframe:pd.DataFrame = None):
        try:
            X = self.__tokenizing_corpus(dframe=dframe)
            '''
            Encoding for ratings
            if rating >= 3 then 1 which means positive
            if rating < 3 then 0 which means negative
            ''' 
            dframe['overall'] = dframe['overall'].apply(lambda x : 1 if x > 3 else 0)
            y = dframe['overall']
            return X,y
        except Exception as e:
            raise Exception(e)