from preprocessing.preprocessor import Preprocessor
from modeling.model import RiskModel
from decision.decision_engine import DecisionEngine
from inference.predictor import RiskPredictor


# Load artifacts
model = RiskModel()
model.load()

preprocessor = Preprocessor()
preprocessor.load()

engine = DecisionEngine(threshold=0.6)

predictor = RiskPredictor(model, preprocessor, engine)

# Example customer
customer = [
    32,
    98000,
    20000,
    720,
    6,
    0.30
]

prob, decision = predictor.predict_customer(customer)

print("\nCustomer Risk Prediction")
print(f"Probability of Default: {prob:.3f}")
print(f"Decision: {decision}")

if decision == "LOW RISK":
    print("Loan Status: APPROVED")
else:
    print("Loan Status: REJECTED")