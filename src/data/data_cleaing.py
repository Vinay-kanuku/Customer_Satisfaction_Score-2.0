from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from scipy import stats
from sklearn.impute import SimpleImputer
from typing import Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class DataPreprocessor(ABC):
    """
    This is an abstract class used for data preprocessing
    """
    @abstractmethod
    def preprocess_data(self, train_df: pd.DataFrame, test_df:pd.DataFrame) -> pd.DataFrame:
        """
        This method is used to preprocess the data
        Args: train_df test_df (raw data.. )
        Returns: df (pd.DataFrame): The preprocessed data.
        """
        pass  

class DropFeaturesWithHighMissingValues(DataPreprocessor):
    """ 
    This class is used to drop features with high missing values in the data
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold 

    def preprocess_data(self, 
                        train_df:pd.DataFrame, 
                        test_df:pd.DataFrame) -> pd.DataFrame:
        """
        This method drops features with high missing values
        Args: train_df test_df (raw data.. )
        Returns: df (pd.DataFrame): The preprocessed data.
        """
        try:
            num_cols = train_df.select_dtypes("int64", "float64")
            cols_to_drop = num_cols[num_cols.isnull().mean() > self.threshold]
            logging.info(f"Dropping features with high missing values. Dropped columns: {cols_to_drop.tolist()}")
            return train_df.drop(columns=cols_to_drop.index.tolist()), test_df.drop(columns=cols_to_drop.index.tolist())
        
        except Exception as e:
            logging.error(f"Failed to preprocess data. Error: {e}")
            raise e
       
class ImputeNumericalMssingValues(DataPreprocessor):
    """ 
    This class is used to impute numerical missing values in the data
    """


    def preprocess_data(self, train_df, test_df):
        """
        This method imputes missing values in the data
        Args: df (pd.DataFrame): The raw data.
        Returns: df (pd.DataFrame): The preprocessed data.
        """
        num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer.fit(train_df[num_cols])
        transformed_train_df = imputer.transform(train_df[num_cols])
        transformed_train_df = pd.DataFrame(transformed_train_df, columns=num_cols, index=train_df.index)
        # print(transformed_train_df.head())
        transformed_test_df = imputer.transform(test_df[num_cols])
        transformed_test_df = pd.DataFrame(transformed_test_df, columns=num_cols, index=test_df.index)
        logging.info(f"Imputed missing values. Training set shape: {transformed_train_df.shape}, Test set shape: {transformed_test_df.shape}")
        return transformed_train_df, transformed_test_df
 
class RemoveOutliers:
    """Strategy for removing outliers using various statistical methods."""

    def __init__(
        self,
        method: str = "zscore",
        threshold: float = 3.0,
        columns: Optional[List[str]] = None,
    ):
        """
        Initialize outlier removal strategy.

        Args:
            method (str): Method to use ('zscore', 'iqr', 'percentile')
            threshold (float): Threshold for z-score method
            columns (List[str], optional): Specific columns to check for outliers
        """
        self.method = method
        self.threshold = threshold
        self.columns = columns

    def preprocess_data(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove outliers from the training DataFrame.

        Args:
            train_df (pd.DataFrame): Training DataFrame
            test_df (pd.DataFrame): Test DataFrame (outliers are not removed here)

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Processed training and test DataFrames
        """
        try:
            # Ensure we work on a copy of the training data
            train_df_copy = train_df.copy()
            rows_before = len(train_df_copy)

            # If no columns specified, use all numerical columns
            if self.columns is None:
                self.columns = train_df_copy.select_dtypes(
                    include=["int64", "float64"]
                ).columns.tolist()

            # Remove outliers from the training set only
            if self.method == "zscore":
                train_df_copy = self._remove_zscore_outliers(train_df_copy)
            elif self.method == "iqr":
                train_df_copy = self._remove_iqr_outliers(train_df_copy)
            elif self.method == "percentile":
                train_df_copy = self._remove_percentile_outliers(train_df_copy)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Log the number of rows removed
            rows_removed = rows_before - len(train_df_copy)
            logger.info(
                f"Removed {rows_removed} rows containing outliers using {self.method} method"
            )

            # Return processed training data and untouched test data
            return train_df_copy, test_df

        except Exception as e:
            logger.error(f"Error in outlier removal: {str(e)}")
            raise

    def _remove_zscore_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        for column in self.columns:
            z_scores = np.abs(stats.zscore(df[column]))
            df = df[z_scores < self.threshold]
        return df

    def _remove_iqr_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        for column in self.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[
                ~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))
            ]
        return df

    def _remove_percentile_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using percentile method."""
        for column in self.columns:
            lower = df[column].quantile(0.01)
            upper = df[column].quantile(0.99)
            df = df[(df[column] >= lower) & (df[column] <= upper)]
        return df
class Preprocessor:
    def __init__(self, 
                 train_df:pd.DataFrame,
                 test_df:pd.DataFrame,
                 strategy:DataPreprocessor):
        self.train_df = train_df
        self.test_df = test_df
        self.strategy = strategy

    def set_strategy(self, strategy:DataPreprocessor):
        self.strategy = strategy

    def preprocess(self) -> pd.DataFrame:
        return self.strategy.preprocess_data(self.train_df, self.test_df)
     
if __name__ == "__main__":
    path1 = "/home/vinay/code/Machine_Learning/customer_satisfaction_score/data/train.csv"
    path2 = "/home/vinay/code/Machine_Learning/customer_satisfaction_score/data/test.csv"
    train_df = pd.read_csv(path1)
    test_df = pd.read_csv(path2)
    # print(train_df.isnull().sum())
    # print(test_df.isnull().sum())
    startegy = ImputeNumericalMssingValues()
    imputer = Preprocessor(train_df,test_df, startegy)
    train_df, test_df = imputer.preprocess()
    startegy = RemoveOutliers()
    imputer.set_strategy(startegy)
    train_df, test_df = imputer.preprocess()
    
    # print(train_df.isnull().sum())
    # print(test_df.isnull().sum())


 
        
 