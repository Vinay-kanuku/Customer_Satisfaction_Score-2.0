
from abc import ABC, abstractmethod
import logging 
import pandas as  pd 
import numbers as np 
import os 

class Ingestor(ABC):
    """
    this is a abstract used to ingest the data 
    """
    @abstractmethod
    def ingest_data(self, path: str) -> pd.DataFrame:
        """
        This methos ingests  the data from a source
        args: path to data 
        returns : pd.DataFrame
        """
        pass 

class CvsInsgestor(Ingestor):
    """ 
    This class is used to ingest data from a CSV file
    """
    def ingest_data(self, path: str) -> pd.DataFrame:
        """
        This method ingests  the data from a CSV file
        args: path to data
        returns : pd.DataFrame
        """
        try:
            df = pd.read_csv(path)
            logging.info(f"reading data from {path} is done.")
            return df
        except FileNotFoundError as e:
            logging.error(f"File not found at {path}. Error: {e}")
            raise e
        

if __name__ == "__main__":
    path = r"/home/vinay/code/Machine_Learning/customer_satisfaction_score/data/olist_customers_dataset.csv"
    csv_ins = CvsInsgestor()
    df = csv_ins.ingest_data(path)
    print(df.head())


    