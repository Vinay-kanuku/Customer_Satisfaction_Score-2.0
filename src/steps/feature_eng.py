from zenml import step
import logging 
import pandas as pd
from pydantic import ConfigDict 

@step 
def feature_engineering(df:pd.DataFrame)-> pd.DataFrame:
    """
    This function performs feature engineering.
    Returns: None
    """
    return df
