from text_preprocessing import text_preprocessing
from model_train_eval import model_trainer,model_eval
import pandas as pd
from tqdm import tqdm

def main():
    print("Hello from sentiment-analysis-text-encoding!")


if __name__ == "__main__":
    tp = text_preprocessing("C:\\Aditya\\Generative AI\\Gen AI Codes\\Sentiment_Analysis_Text_Encoding\\amazon_reviews.csv",'csv')
    collist = ['Unnamed: 0','reviewerName','reviewTime','day_diff','helpful_yes', 'helpful_no', 'total_vote','score_pos_neg_diff', 'score_average_rating', 'wilson_lower_bound']
    df = tp.text_cleaning(unwanted_cols=collist)
    X,y = tp.features_setting(dframe=df)
    mt = model_trainer()
    encoders = ['BOW','TFIDF']
    evals_list = []
    for i in tqdm(range(len(encoders))):
        model,X_test,y_test,vect = mt.train_model(X,y,encoders[i])
        mte = model_eval()
        model_evals = mte.model_eval(model,X_test,y_test,encoders[i])
        evals_list.append(model_evals)
    evals_df = pd.DataFrame(evals_list)
    print(evals_df)
    

