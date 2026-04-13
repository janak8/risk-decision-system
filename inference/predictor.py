# inference/predictor.py

import numpy as np


class RiskPredictor:

    def __init__(self, model, preprocessor, decision_engine):
        self.model = model
        self.preprocessor = preprocessor
        self.decision_engine = decision_engine

    def predict_customer(self, customer_data):

        # Convert input to array
        X = np.array(customer_data).reshape(1, -1)

        # Apply same preprocessing
        X_scaled = self.preprocessor.transform(X)

        # Get probability
        probability = self.model.predict_proba(X_scaled)[0]

        # Get decision
        decision = self.decision_engine.make_decision(probability)

        return probability, decision