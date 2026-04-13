class ExplainabilityEngine:
    def __init__(self, model):
        self.model = model.model  # sklearn model

    def explain(self, features, feature_names):
        """
        features: scaled feature array (1D)
        feature_names: list of feature names
        """

        coefs = self.model.coef_[0]

        explanations = []

        for name, value, coef in zip(feature_names, features, coefs):
            impact = value * coef

            explanations.append({
                "feature": name,
                "value": float(value),
                "coefficient": float(coef),
                "impact": float(impact)
            })

        # Sort by absolute impact (most important first)
        explanations = sorted(explanations, key=lambda x: abs(x["impact"]), reverse=True)

        return explanations