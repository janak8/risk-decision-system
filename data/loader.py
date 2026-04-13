# data/loader.py

import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, file_path, target_column="default"):
        """
        Initialize the DataLoader.

        Args:
            file_path (str): Path to CSV file
            target_column (str): Name of the target column
        """
        self.file_path = file_path
        self.target_column = target_column
        self.df = None
        self.X = None
        self.y = None

    def load_data(self):
        """Load CSV into dataframe and separate features and target."""
        self.df = pd.read_csv(self.file_path)
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")

        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]
        print(f"[DataLoader] Loaded data: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

    def train_test_split(self, test_size=0.2, random_state=42, stratify=True):
        """
        Split data into train and test sets.

        Args:
            test_size (float): Fraction of data for testing
            random_state (int): Random seed
            stratify (bool): Whether to stratify based on target

        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        stratify_data = self.y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=stratify_data
        )
        print(f"[DataLoader] Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} rows")
        return X_train, X_test, y_train, y_test

    def get_features_and_target(self):
        """Return features and target for external use."""
        if self.X is None or self.y is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self.X, self.y
