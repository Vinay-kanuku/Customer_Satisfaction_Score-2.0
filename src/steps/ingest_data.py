import logging 
from zenml import step 
import pandas as pd 

from pydantic import ConfigDict 

@step 
def ingest_df(path:str) -> None:
    """
    This  function is repsonsible for ingestion step.. 
    args: path 
    retunr: None 
    """
    df = pd.read_csv(path)
    return df 
  