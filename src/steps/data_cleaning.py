import logging 
from zenml import step
import pandas as pd
 
@step  
def clean_df(df:pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the data.
    Args: df (pandas DataFrame): The raw data.
    Returns: df (pandas DataFrame): The cleaned data.
    """
    return df

