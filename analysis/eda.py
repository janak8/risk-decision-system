# analysis/eda.py

import pandas as pd


class RiskDataEDA:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("\n[INFO] Dataset Loaded Successfully")

    def basic_info(self):
        print("\n===== BASIC INFO =====")
        print(f"Shape: {self.df.shape}")
        print("\nColumns:")
        print(self.df.columns.tolist())
        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def target_distribution(self):
        print("\n===== TARGET DISTRIBUTION =====")
        print(self.df["default"].value_counts(normalize=True))

    def feature_summary(self):
        print("\n===== FEATURE SUMMARY =====")
        print(self.df.describe())

    def correlation_with_target(self):
        print("\n===== CORRELATION WITH TARGET =====")
        corr = self.df.corr()["default"].sort_values(ascending=False)
        print(corr)

    def run_full_analysis(self):
        self.load_data()
        self.basic_info()
        self.target_distribution()
        self.feature_summary()
        self.correlation_with_target()
