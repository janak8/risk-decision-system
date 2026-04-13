from pydantic import BaseModel


class CustomerData(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_score: int
    years_employed: int
    debt_to_income: float