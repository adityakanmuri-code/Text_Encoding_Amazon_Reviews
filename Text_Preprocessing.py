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
import sys


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
            df = self.__remove_unwanted_cols(unwanted_cols)
            return df
        except Exception as e:
            raise Exception(e)
    
    def __remove_unwanted_cols(self,dframe:pd.DataFrame = None,unwanted_cols:list = None):
        try:
            return dframe.drop(columns=unwanted_cols)
        except Exception as e:
            raise Exception(e)
        
    