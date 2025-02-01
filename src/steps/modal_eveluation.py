from zenml import step
from sklearn.metrics import accuracy_score
import pandas as pd 

from pydantic import ConfigDict 

@step 
def model_eval(model, df: pd.DataFrame) -> dict:
    """
    Evaluates the trained model.
    """
    X = df.drop(columns=["target"])
    y_true = df["target"]
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y_true, y_pred)
    return {"accuracy": accuracy}
