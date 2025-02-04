import logging 
from zenml import step 
import pandas as pd 

 

@step  
def ingest_df(path:str) -> pd.DataFrame:
    """
    This  function is repsonsible for ingestion step.. 
    args: path 
    retunr: None 
    """
    df = pd.read_csv(path)
    return df 
  