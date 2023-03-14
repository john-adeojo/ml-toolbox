import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import unittest


class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5, 6],
            'B': ['a', 'b', 'a', 'b', 'a', 'b'],
            'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'target': [0, 1, 0, 1, 0, 1]
        })

    def test_split_data(self):
        """
        Test that split_data splits the DataFrame into training and test sets with the correct sizes and class balance.
        """
        fe = FeatureEngineer()
        train_df, test_df = fe.split_data(self.df, 'target', test_size=0.2, random_state=42)
        self.assertEqual(len(train_df), 4)
        self.assertEqual(len(test_df), 2)
        self.assertEqual(len(train_df[train_df['target'] == 0]), 2)
        self.assertEqual(len(train_df[train_df['target'] == 1]), 2)
        self.assertEqual(len(test_df[test_df['target'] == 0]), 1)
        self.assertEqual(len(test_df[test_df['target'] == 1]), 1)

    def test_impute_missing_values(self):
        """
        Test that impute_missing_values imputes missing values in the DataFrame using the specified strategy.
        """
        fe = FeatureEngineer()
        imputed_df = fe.impute_missing_values(self.df, strategy='mean', columns=['A'])
        self.assertAlmostEqual(imputed_df['A'][2], 3.0, places=1)

    def test_feature_scale(self):
        """
        Test that feature_scale scales the specified feature columns in the DataFrame using the specified method.
        """
        fe = FeatureEngineer()
        scaled_df = fe.feature_scale(self.df, feature_columns=['C'])
        self.assertAlmostEqual(scaled_df['C'][0], -1.2247, places=4)

    def test_one_hot_encode(self):
        """
        Test that one_hot_encode performs one-hot encoding on the specified categorical columns in the DataFrame.
        """
        fe = FeatureEngineer()
        encoded_df = fe.one_hot_encode(self.df, categorical_columns=['B'])
        self.assertEqual(encoded_df.shape[1], 3)
        self.assertEqual(encoded_df.iloc[0, 0], 1)

    def test_reduce_dimensionality(self):
        """
        Test that reduce_dimensionality performs dimensionality reduction on the DataFrame using the specified method.
        """
        fe = FeatureEngineer()
        pca_df = fe.reduce_dimensionality(self.df, method='pca', n_components=2)
        self.assertEqual(pca_df.shape[1], 2)

    def test_engineer_features(self):
        """
        Test that engineer_features performs all feature engineering steps correctly and returns training and test sets.
        """
        fe = FeatureEngineer()
        train_df, test_df = fe.engineer_features(self.df, target_column='target',
                                                  impute_strategy='mean',
                                                  scaler_features=['C'],
                                                  one_hot_encode_columns=['B'],
                                                  dim_reduction_method='pca',
                                                  dim_reduction_components=2,
                                                  test_size=0.2,
                                                  random_state=42)
        # Train a simple logistic regression model
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.assertAlmostEqual(accuracy, 1.0, places=2)

    def test_impute_missing_values_with_no_missing_values(self):
        """
        Test that impute_missing_values returns the original DataFrame when there are no missing values.
        """
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })

        fe = FeatureEngineer()
        imputed_df = fe.impute_missing_values(df, strategy='mean', columns=['A'])

        self.assertTrue(imputed_df.equals(df))

    def test_impute_missing_values_with_all_missing_values(self):
        """
        Test that impute_missing_values returns a DataFrame with all missing values when all values are missing.
        """
        df = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': ['a', 'b', 'c']
        })

        fe = FeatureEngineer()
        imputed_df = fe.impute_missing_values(df, strategy='mean', columns=['A'])

        self.assertTrue(imputed_df['A'].isnull().all())

    def test_feature_scale_with_single_value_column(self):
        """
        Test that feature_scale returns the original DataFrame when scaling a single-value column.
        """
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })

        fe = FeatureEngineer()
        scaled_df = fe.feature_scale(df, feature_columns=['A'])

        self.assertTrue(scaled_df.equals(df))

    def test_feature_scale_with_all_same_values(self):
        """
        Test that feature_scale returns a DataFrame with all values scaled to zero when all values are the same.
        """
        df = pd.DataFrame({
            'A': [1, 1, 1],
            'B': ['a', 'b', 'c']
        })

        fe = FeatureEngineer()
        scaled_df = fe.feature_scale(df, feature_columns=['A'])

        self.assertAlmostEqual(scaled_df['A'][0], 0.0, places=2)
        self.assertAlmostEqual(scaled_df['A'][1], 0.0, places=2)
        self.assertAlmostEqual(scaled_df['A'][2], 0.0, places=2)
