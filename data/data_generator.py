# data/data_generator.py

import numpy as np
import pandas as pd


class CreditDataGenerator:
    def __init__(self, n_samples=5000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(self.random_state)

    def generate(self):
        # Feature generation (semi-realistic distributions)

        age = np.random.normal(40, 10, self.n_samples).clip(21, 65)
        income = np.random.normal(60000, 20000, self.n_samples).clip(20000, 150000)
        loan_amount = np.random.normal(25000, 15000, self.n_samples).clip(2000, 80000)
        credit_score = np.random.normal(680, 50, self.n_samples).clip(300, 850)
        years_employed = np.random.normal(7, 5, self.n_samples).clip(0, 40)
        debt_to_income = np.random.normal(0.35, 0.15, self.n_samples).clip(0.05, 0.9)

        # Hidden risk formula (true underlying signal)
        logit = (
            -0.00003 * income +
            0.00004 * loan_amount +
            -0.005 * credit_score +
            -0.03 * years_employed +
            3.0 * debt_to_income +
            -0.01 * age +
            5  # bias term
        )

        probability = 1 / (1 + np.exp(-logit))
        default = np.random.binomial(1, probability)

        df = pd.DataFrame({
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "years_employed": years_employed,
            "debt_to_income": debt_to_income,
            "default": default
        })

        return df

    def save(self, path="data/credit_risk_dataset.csv"):
        df = self.generate()
        df.to_csv(path, index=False)
        print(f"Dataset saved to {path}")
