 
import pandas 
import logging 
from zenml import pipeline
from src.steps.ingest_data import ingest_df
from src.steps.data_cleaning import clean_df
from src.steps.feature_eng import feature_engineering
from src.steps.modal_traning import train_model
from src.steps.modal_eveluation import model_eval
logging.basicConfig(level=logging.INFO)
 

@pipeline 
def training_pipeline(path: str):
    df = ingest_df(path)
    df = clean_df(df)
    df = feature_engineering(df)
 


if __name__ == "__main__":
    path = r"/home/vinay/code/Machine_Learning/customer_satisfaction_score/data/olist_customers_dataset.csv"
    training_pipeline(path)
 