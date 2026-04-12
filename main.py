from text_preprocessing import text_preprocessing
from model_train_eval import model_trainer
import pandas as pd

def main():
    print("Hello from sentiment-analysis-text-encoding!")


if __name__ == "__main__":
    tp = text_preprocessing("C:\\Aditya\\Generative AI\\Gen AI Codes\\Sentiment_Analysis_Text_Encoding\\amazon_reviews.csv",'csv')
    collist = ['Unnamed: 0','reviewerName','reviewTime','day_diff','helpful_yes', 'helpful_no', 'total_vote','score_pos_neg_diff', 'score_average_rating', 'wilson_lower_bound']
    df = tp.text_cleaning(unwanted_cols=collist)
    X,y = tp.features_setting(dframe=df)
    mt = model_trainer()
    

