 
import pandas 
import logging 
from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.data_cleaning import clean_df
from steps.feature_eng import feature_engineering
from steps.modal_eveluation import model_eval 
from steps.modal_traning import train_model 
 
logging.basicConfig(level=logging.INFO)
 

@pipeline 
def train_pipeline(path: str):
    df = ingest_df(path)
    df = clean_df(df)
    df = feature_engineering(df)
    modal = train_model(df)
    # metrics = train_model(modal)
 


# if __name__ == "__main__":
#     path = r"/home/vinay/code/Machine_Learning/customer_satisfaction_score/data/olist_customers_dataset.csv"
#     train_pipeline(path)
 