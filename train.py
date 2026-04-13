from data.loader import DataLoader
from preprocessing.preprocessor import Preprocessor
from modeling.model import RiskModel
from modeling.trainer import ModelTrainer
from decision.decision_engine import DecisionEngine


if __name__ == "__main__":

    loader = DataLoader("data/credit_risk_dataset.csv")
    loader.load_data()

    X_train, X_test, y_train, y_test = loader.train_test_split()

    preprocessor = Preprocessor()

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    model = RiskModel()

    trainer = ModelTrainer(model)
    trainer.train(X_train_scaled, y_train)

    # save artifacts
    model.save()
    preprocessor.save()

    trainer.evaluate(X_test_scaled, y_test)

    # Get probabilities
    probabilities = model.predict_proba(X_test_scaled)

    # Decision system
    engine = DecisionEngine(threshold=0.6)

    decisions = engine.evaluate_predictions(probabilities[:10])

    print("\nSample Decisions:")
    for p, d in zip(probabilities[:10], decisions):
        print(f"Probability: {p:.3f} → {d}")

from inference.predictor import RiskPredictor

# Create predictor
predictor = RiskPredictor(model, preprocessor, engine)

# Example customer
customer = [
    32,      # age
    65000,   # income
    20000,   # loan_amount
    720,     # credit_score
    20,       # years_employed
    0.30     # debt_to_income
]

prob, decision = predictor.predict_customer(customer)

print("\nCustomer Risk Prediction")
print(f"Probability of Default: {prob:.3f}")
print(f"Decision: {decision}")

if decision == "LOW RISK":
    print("Loan Status: APPROVED")
else:
    print("Loan Status: REJECTED")


# explainability
from explainability.coefficient_explainer import CoefficientExplainer

feature_names = X_train.columns

explainer = CoefficientExplainer(model, feature_names)

explainer.print_explanation()