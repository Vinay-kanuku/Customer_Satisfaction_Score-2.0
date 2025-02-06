import logging 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import os 

class SplitData:
    def __init__(self,df, train_path, test_path):
        self.df = df 
        self.train_path = train_path 
        self.test_path = test_path

    def split_data(self, target):
        """
        Splits the data into training and testing sets.
        
        Args:
            target (str): The name of the target variable.
        
        Returns:
            None
        """
        # Check if train and test files already exist
        if os.path.exists(self.train_path) and os.path.exists(self.test_path):
            logging.info("Train and test files already exist. Skipping data splitting.")
            return
        X = self.df.drop(columns=[target])
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Splitting data into train and test sets. Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        self.save_data(X_train, y_train, self.train_path)
        self.save_data(X_test, y_test, self.test_path)

    def save_data(self, X, y, path):
        """
        Saves the split data as CSV files.
        
        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series): The target data.
            path (str): The path to save the data.
        
        Returns:
            None
        """
        try:
            df = pd.concat([X, y], axis=1)
            df.to_csv(path, index=False)
            logging.info(f"Data saved to {path}")

        except FileExistsError as e:
            logging.error(f"Failed to save data to {path}. Error: {e}")
            raise e
        except FileNotFoundError as e:
            logging.error(f"Failed to save data to {path}. Error: {e}")
            raise e
        

if __name__ == "__main__":
    path = r"/home/vinay/code/Machine_Learning/customer_satisfaction_score/data/olist_customers_dataset.csv"
    df = pd.read_csv(path)
    split_data = SplitData(df, train_path="/home/vinay/code/Machine_Learning/customer_satisfaction_score/data/train.csv", test_path="/home/vinay/code/Machine_Learning/customer_satisfaction_score/data/test.csv")
    split_data.split_data("review_score") 

    
        
 
