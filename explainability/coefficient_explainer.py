import pandas as pd


class CoefficientExplainer:

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def get_coefficients(self):

        coefficients = self.model.model.coef_[0]

        df = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": coefficients
        })

        df["impact"] = df["coefficient"].apply(
            lambda x: "Increase Risk" if x > 0 else "Decrease Risk"
        )

        df = df.sort_values(by="coefficient", ascending=False)

        return df

    def print_explanation(self):

        df = self.get_coefficients()

        print("\nModel Risk Factors:\n")

        for _, row in df.iterrows():

            feature = row["feature"]
            coef = row["coefficient"]
            impact = row["impact"]

            print(f"{feature:15} {coef: .3f} → {impact}")