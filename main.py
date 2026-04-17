from text_preprocessing import text_preprocessing
from model_train_eval import model_trainer
import pandas as pd
from tqdm import tqdm
from scrapper import run_scraper
import pickle

def main():
    print("Hello from sentiment-analysis-text-encoding!")


if __name__ == "__main__":
   #Step 1 : Performing Model Training on the Kaggle Dataset
    tp = text_preprocessing("C:\\Aditya\\Generative AI\\Gen AI Codes\\Sentiment_Analysis_Text_Encoding\\amazon_reviews.csv",'csv')
    collist = ['Unnamed: 0','reviewerName','reviewTime','day_diff','helpful_yes', 'helpful_no', 'total_vote','score_pos_neg_diff', 'score_average_rating', 'wilson_lower_bound']
    df = tp.text_cleaning(unwanted_cols=collist)
    X,y = tp.features_setting(dframe=df)

    trainer = model_trainer()
    model,model_name,best_score,vectorizer = trainer.train_model(X,y,encode_tech='TFIDF')
    with open("sentimment_model.pkl","wb") as f:
        pickle.dump(model,f)
    
    with open("vectorizer.pkl","wb") as f:
        pickle.dump(vectorizer,f)


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



    tp_scrape = text_preprocessing("C:\\Aditya\\Generative AI\\Gen AI Codes\\Sentiment_Analysis_Text_Encoding\\amazon_reviews_scrapped.csv","csv")
    df_scraped = tp_scrape.text_cleaning(unwanted_cols=None)

    X_new,_ = tp_scrape.features_setting(dframe=df_scraped)

    X_vectorized = vectorizer.transform(X_new)

    predictions = model.predict(X_vectorized)

    df_scraped['predicted_sentiment'] = predictions

    print(df_scraped[['reviewText', 'predicted_sentiment']].head())







    

