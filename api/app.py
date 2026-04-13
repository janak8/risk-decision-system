from fastapi import FastAPI

from preprocessing.preprocessor import Preprocessor
from modeling.model import RiskModel
from decision.decision_engine import DecisionEngine
from inference.predictor import RiskPredictor

from api.schema import CustomerData


app = FastAPI(title="Loan Risk Decision API")


# Load artifacts
model = RiskModel()
model.load()

preprocessor = Preprocessor()
preprocessor.load()

engine = DecisionEngine(threshold=0.6)

predictor = RiskPredictor(model, preprocessor, engine)


@app.get("/")
def home():
    return {"message": "Risk Decision System API"}


@app.post("/predict")
def predict(customer: CustomerData):

    data = [
        customer.age,
        customer.income,
        customer.loan_amount,
        customer.credit_score,
        customer.years_employed,
        customer.debt_to_income
    ]

    probability, decision = predictor.predict_customer(data)

    return {
        "default_probability": float(probability),
        "risk_level": decision,
        "loan_status": "APPROVED" if decision == "LOW RISK" else "REJECTED"
    }