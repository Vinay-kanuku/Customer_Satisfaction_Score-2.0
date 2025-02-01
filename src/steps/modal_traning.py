from zenml import step
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pydantic import ConfigDict 

@step 
def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Trains a machine learning model.
    """
    X = df.drop(columns=["target"])
    y = df["target"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
