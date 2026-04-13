# decision/decision_engine.py

class DecisionEngine:
    def __init__(self, threshold=0.6):
        """
        threshold: probability cutoff for high risk
        """
        self.threshold = threshold

    def make_decision(self, probability):
        """
        Convert probability into risk decision
        """
        if probability >= self.threshold:
            return "HIGH RISK"
        else:
            return "LOW RISK"

    def evaluate_predictions(self, probabilities):
        """
        Apply decision rule to multiple predictions
        """
        decisions = []

        for p in probabilities:
            decision = self.make_decision(p)
            decisions.append(decision)

        return decisions