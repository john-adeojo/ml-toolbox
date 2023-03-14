import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA


class FeatureEngineer:
    """
    A class for performing common feature engineering tasks on a Pandas DataFrame.

    Parameters:
        None

    Returns:
        None
    """

    def __init__(self):
        pass

    def split_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Split a Pandas DataFrame into training and test sets, with stratification based on the target column.

        Parameters:
            df (pd.DataFrame): The DataFrame to be split.
            target_column (str): The name of the target column.
            test_size (float): The proportion of the data to use for the test set. Default is 0.2.
            random_state (int): The random seed to use for reproducibility. Default is 42.

        Returns:
            tuple: A tuple containing the training and test sets as DataFrames.
        """
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        return train_df, test_df

    def impute_missing_values(self, df: pd.DataFrame, strategy: str = "mean", columns: list = None) -> pd.DataFrame:
        """
        Impute missing values in a Pandas DataFrame using the specified strategy.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the missing values to be imputed.
            strategy (str): The imputation strategy to use. Can be "mean", "median", "most_frequent", or a constant value. Default is "mean".
            columns (list): A list of column names to apply the imputation to. If None, imputes all columns. Default is None.

        Returns:
            pd.DataFrame: A new DataFrame with the missing values imputed.
        """
        imputer = SimpleImputer(strategy=strategy)
        if columns is None:
            imputed_array = imputer.fit_transform(df)
            imputed_df = pd.DataFrame(imputed_array, columns=df.columns)
        else:
            imputed_array = imputer.fit_transform(df[columns])
            imputed_df = df.copy()
            imputed_df[columns] = imputed_array
        return imputed_df

    def feature_scale(self, df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        """
        Scale the specified feature columns of a Pandas DataFrame using the StandardScaler.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the features to be scaled.
            feature_columns (list): A list of column names for the features to be scaled.

        Returns:
            pd.DataFrame: A new DataFrame with the scaled features.
        """
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[feature_columns])
        scaled_df = df.copy()
        for i, col in enumerate(feature_columns):
            scaled_df[col] = scaled_features[:, i]
        return scaled_df



    def one_hot_encode(self, df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
        """
        Perform one-hot encoding on the specified categorical columns of a Pandas DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the categorical features to be one-hot encoded.
            categorical_columns (list): A list of column names for the categorical features to be one-hot encoded.

        Returns:
            pd.DataFrame: A new DataFrame with the one-hot encoded features.
        """
        encoder = OneHotEncoder()
        encoded_array = encoder.fit_transform(df[categorical_columns]).toarray()
        feature_names = encoder.get_feature_names(categorical_columns)
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names)
        return encoded_df

    def reduce_dimensionality(self, df: pd.DataFrame, method: str = "pca", n_components: int = None) -> pd.DataFrame:
        """
        Reduce the dimensionality of a Pandas DataFrame using the specified method.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the features to be reduced.
            method (str): The dimensionality reduction method to use. Can be "pca" (for principal component analysis) or "lda" (for linear discriminant analysis). Default is "pca".
            n_components (int): The number of components to retain after dimensionality reduction. Only used if method is "pca". Default is None.

        Returns:
            pd.DataFrame: A new DataFrame with the dimensionality-reduced features.
        """
        if method == "pca":
            pca = PCA(n_components=n_components)
            pca_array = pca.fit_transform(df)
            pca_df = pd.DataFrame(pca_array, columns=[f"PC{i}" for i in range(1, n_components+1)])
            return pca_df
        elif method == "lda":
            pass  # Add code for LDA dimensionality reduction here
        else:
            raise ValueError("Invalid dimensionality reduction method specified. Must be 'pca' or 'lda'.")

    def engineer_features(self, df: pd.DataFrame, target_column: str, impute_strategy: str = "mean",
                          scaler_features: list = None, one_hot_encode_columns: list = None,
                          dim_reduction_method: str = None, dim_reduction_components: int = None,
                          test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Apply a series of feature engineering transformations to a Pandas DataFrame, split it into training and test sets, and return the transformed data as well as the training and test sets.

        Parameters:
            df (pd.DataFrame): The DataFrame to be feature engineered.
            target_column (str): The name of the target column.
            impute_strategy (str): The imputation strategy to use. Can be "mean", "median", "most_frequent", or a constant value. Default is "mean".
            scaler_features (list): A list of column names for the features to be scaled using the StandardScaler. If None, does not perform feature scaling. Default is None.
            one_hot_encode_columns (list): A list of column names for the categorical features to be one-hot encoded. If None, does not perform one-hot encoding. Default is None.
            dim_reduction_method (str): The dimensionality reduction method to use. Can be "pca" (for principal component analysis) or "lda" (for linear discriminant analysis). If None, does not perform dimensionality reduction. Default is None.
            dim_reduction_components (int): The number of components to retain after dimensionality reduction. Only used if dim_reduction_method is "pca". Default is None.
            test_size (float): The proportion of the data to use for the test set. Default is 0.2.
            random_state (int): The random seed to use for reproducibility. Default is 42.

        Returns:
            tuple: A tuple containing the transformed DataFrame, training set, and test set.
        """
        # Split data into training and test sets
        train_df, test_df = self.split_data(df, target_column, test_size=test_size, random_state=random_state)

        # Impute missing values
        train_df = self.impute_missing_values(train_df, strategy=impute_strategy)
        test_df = self.impute_missing_values(test_df, strategy=impute_strategy)

        # Scale features
        if scaler_features is not None:
            train_df = self.feature_scale(train_df, scaler_features)
            test_df = self.feature_scale(test_df, scaler_features)

        # One-hot encode categorical features
        if one_hot_encode_columns is not None:
            train_df = self.one_hot_encode(train_df, one_hot_encode_columns)
            test_df = self.one_hot_encode(test_df, one_hot_encode_columns)

        # Reduce dimensionality
        if dim_reduction_method is not None:
            if dim_reduction_method == "pca":
                train_df = self.reduce_dimensionality(train_df, method="pca", n_components=dim_reduction_components)
                test_df = self.reduce_dimensionality(test_df, method="pca", n_components=dim_reduction_components)
            elif dim_reduction_method == "lda":
                pass  # Add code for LDA dimensionality reduction here
            else:
                raise ValueError("Invalid dimensionality reduction method specified. Must be 'pca' or 'lda'.")

        return train_df, test_df




