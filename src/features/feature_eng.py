from abc import ABC, abstractmethod
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from textblob import TextBlob


class FeatureEng(ABC):
    """
    This is an abstract class used for feature engineering.
    """
    @abstractmethod
    def feature_engineering(self, 
                            train_df: pd.DataFrame,
                            test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        This method is used to perform feature engineering.
        Args:
            train_df (pd.DataFrame): The training data to be processed.
            test_df (pd.DataFrame): The test data to be processed.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The processed training and test data.
        """
        pass


class RemoveObviousFeature(FeatureEng):
    """ 
    This class is used to remove obvious features from the data.
    """
    def feature_engineering(self, 
                            train_df: pd.DataFrame,
                            test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove obvious features from the data.
        Args:
            train_df (pd.DataFrame): Training DataFrame.
            test_df (pd.DataFrame): Test DataFrame.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Processed training and test DataFrames.
        """
        OBVIOUS_FEATURES = [
            'order_id',   
            'customer_id',    
            'customer_unique_id',    
            'product_id',    
            'seller_id',    
            'product_photos_qty', 
            'customer_zip_code_prefix',
            'order_item_id',
            'product_category_name',
        ]
        logging.info(f"Removing obvious features: {OBVIOUS_FEATURES}")
        train_df = train_df.drop(columns=OBVIOUS_FEATURES, axis=1, errors='ignore')
        test_df = test_df.drop(columns=OBVIOUS_FEATURES, axis=1, errors='ignore')
        logging.info(f"Training set shape after removing obvious features: {train_df.shape}, Test set shape after removing obvious features: {test_df.shape}")
        return train_df, test_df


class FeatureCreation(FeatureEng):
    """ 
    This class is used to create new features from existing ones.
    """
    def feature_engineering(self, 
                            train_df: pd.DataFrame, 
                            test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create new features for both training and test datasets.
        Args:
            train_df (pd.DataFrame): Training DataFrame.
            test_df (pd.DataFrame): Test DataFrame.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Processed training and test DataFrames.
        """
        # Apply feature creation to both train and test datasets
        logging.info("Creating new features for the training dataset.")
        train_df = self._create_features(train_df)
        logging.info("Creating new features for the test dataset.")
        test_df = self._create_features(test_df)
        return train_df, test_df

    @staticmethod
    def _create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features for a given DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with new features added.
        """
        # Delivery performance features
        df = FeatureCreation.calculate_delivery_delay(df)
        df = FeatureCreation.calculate_delivery_time(df)
        df = FeatureCreation.is_late_delivery(df)
        # Review sentiment features
        df = FeatureCreation.calculate_sentiment_score(df)
        df = FeatureCreation.classify_sentiment(df)

        # Drop parent features that are no longer necessary
        PARENT_FEATURES_TO_DROP = [
            'review_comment_message',  # Used for sentiment analysis
            'order_purchase_timestamp',  # Used for delivery time calculation
            'order_delivered_customer_date',  # Used for delivery delay and time
            'order_estimated_delivery_date'  # Used for delivery delay
        ]
        df = df.drop(columns=PARENT_FEATURES_TO_DROP, errors='ignore')
        logging.info(f"Dropped parent features: {PARENT_FEATURES_TO_DROP}")
        return df

    @staticmethod
    def calculate_delivery_delay(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate delivery delay (actual delivery date - estimated delivery date).
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with 'delivery_delay' column added.
        """
        if 'order_delivered_customer_date' in df.columns and 'order_estimated_delivery_date' in df.columns:
            df['delivery_delay'] = (
                pd.to_datetime(df['order_delivered_customer_date']) -
                pd.to_datetime(df['order_estimated_delivery_date'])
            ).dt.days
            logging.info("Created 'delivery_delay' feature.")
        else:
            logging.warning("Required columns for 'delivery_delay' are missing.")
        return df

    @staticmethod
    def calculate_delivery_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total delivery time (order placed to order delivered).
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with 'delivery_time' column added.
        """
        if 'order_delivered_customer_date' in df.columns and 'order_purchase_timestamp' in df.columns:
            df['delivery_time'] = (
                pd.to_datetime(df['order_delivered_customer_date']) -
                pd.to_datetime(df['order_purchase_timestamp'])
            ).dt.days
            logging.info("Created 'delivery_time' feature.")
        else:
            logging.warning("Required columns for 'delivery_time' are missing.")
        return df

    @staticmethod
    def is_late_delivery(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary feature indicating whether the delivery was late.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with 'is_late_delivery' column added.
        """
        if 'delivery_delay' in df.columns:
            df['is_late_delivery'] = (df['delivery_delay'] > 0).astype(int)
            logging.info("Created 'is_late_delivery' feature.")
        else:
            logging.warning("Required column 'delivery_delay' is missing.")
        return df

    @staticmethod
    def calculate_sentiment_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment score from review comments.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with 'sentiment_score' column added.
        """
        if 'review_comment_message' in df.columns:
            df['sentiment_score'] = df['review_comment_message'].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0
            )
            logging.info("Created 'sentiment_score' feature.")
        else:
            logging.warning("Required column 'review_comment_message' is missing.")
        return df

    @staticmethod
    def classify_sentiment(df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify sentiment into 'Negative', 'Neutral', or 'Positive'.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with 'sentiment_label' column added.
        """
        if 'sentiment_score' in df.columns:
            df['sentiment_label'] = pd.cut(
                df['sentiment_score'],
                bins=[-1, -0.1, 0.1, 1],
                labels=['Negative', 'Neutral', 'Positive']
            )
            logging.info("Created 'sentiment_label' feature.")
        else:
            logging.warning("Required column 'sentiment_score' is missing.")
        return df


class SelectiveOneHotEncoding(FeatureEng):
    """ 
    This class is used to perform selective one-hot encoding on essential categorical features.
    """
    def feature_engineering(self, 
                            train_df: pd.DataFrame,
                            test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform one-hot encoding on essential categorical features.
        Args:
            train_df (pd.DataFrame): Training DataFrame.
            test_df (pd.DataFrame): Test DataFrame.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Processed training and test DataFrames.
        """
        # Identify essential categorical features
        essential_categorical_features = ['sentiment_label']  # Add more if needed
        categorical_features = [col for col in essential_categorical_features if col in train_df.columns]
        
        if not categorical_features:
            logging.info("No essential categorical features found for one-hot encoding.")
            return train_df, test_df
        
        # Initialize and fit the encoder on training data
        logging.info(f"Performing one-hot encoding on essential categorical features: {categorical_features}")
        try:
            # For scikit-learn >= 1.2
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  
        except TypeError:
            # For scikit-learn < 1.2
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  

        train_encoded = encoder.fit_transform(train_df[categorical_features])
        test_encoded = encoder.transform(test_df[categorical_features])
        
        # Generate new column names for encoded features
        encoded_columns = encoder.get_feature_names_out(categorical_features)
        
        # Convert encoded features back to DataFrames
        train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_columns, index=train_df.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_columns, index=test_df.index)
        
        # Drop original categorical columns and concatenate encoded features
        train_df = pd.concat([train_df.drop(columns=categorical_features), train_encoded_df], axis=1)
        test_df = pd.concat([test_df.drop(columns=categorical_features), test_encoded_df], axis=1)
        
        logging.info(f"One-hot encoding completed. New shape of training data: {train_df.shape}, Test data: {test_df.shape}")
        return train_df, test_df


class FeatureEngineering:
    """ 
    This class is used to orchestrate the feature engineering process.
    """
    def __init__(self, train_df, test_df):
        self.feature_eng = [
            RemoveObviousFeature(),
            FeatureCreation(),
            SelectiveOneHotEncoding()
        ]
        self.train_df = train_df
        self.test_df = test_df

    def process(self):
        for strategy in self.feature_eng:
            self.train_df, self.test_df = strategy.feature_engineering(self.train_df, self.test_df)
            print(self.train_df.shape)
        return self.train_df, self.test_df


if __name__ == "__main__":
    path1 = r"/home/vinay/code/Machine_Learning/customer_satisfaction_score/data/train.csv"
    path2 = r"/home/vinay/code/Machine_Learning/customer_satisfaction_score/data/test.csv"
    train_df = pd.read_csv(path1)
    test_df = pd.read_csv(path2)
    feature_engineer = FeatureEngineering(train_df, test_df)
    train_df, test_df = feature_engineer.process()
    print(train_df.columns)