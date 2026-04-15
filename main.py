from text_preprocessing import text_preprocessing
from model_train_eval import model_trainer,model_eval
import pandas as pd
from tqdm import tqdm
from scrapper import run_scraper

def main():
    print("Hello from sentiment-analysis-text-encoding!")


if __name__ == "__main__":
    search_urls = [
        "https://www.amazon.in/s?k=xbox",
        "https://www.amazon.in/s?k=ac",
        "https://www.amazon.in/s?k=oven",
        "https://www.amazon.in/s?k=apple",
        "https://www.amazon.in/s?k=poco",
        "https://www.amazon.in/s?k=portronics",
        "https://www.amazon.in/s?k=playstation",
        "https://www.amazon.in/s?k=dji+drone"
    ]

    max_reviews = 200
    reviews = run_scraper(search_urls=search_urls,output_file = 'amazon_reviews_scrapped.csv',max_reviews = max_reviews)
    
    tp = text_preprocessing("C:\\Aditya\\Generative AI\\Gen AI Codes\\Sentiment_Analysis_Text_Encoding\\amazon_reviews.csv",'csv')
    collist = ['Unnamed: 0','reviewerName','reviewTime','day_diff','helpful_yes', 'helpful_no', 'total_vote','score_pos_neg_diff', 'score_average_rating', 'wilson_lower_bound']
    df = tp.text_cleaning(unwanted_cols=collist)
    X,y = tp.features_setting(dframe=df)

