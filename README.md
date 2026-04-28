# 🏦 Risk Decision System (ML API)

This project is a machine learning-powered API that predicts whether a loan application is **high risk or low risk** based on user financial data.

---

## 🚀 Features

* Logistic Regression model for risk prediction
* REST API built using FastAPI
* Data preprocessing with feature scaling
* Model persistence using joblib
* Dockerized for easy deployment

---

## 🧠 How it Works

User Input → Preprocessing → Model Prediction → Risk Decision

---

## 🛠 Tech Stack

* Python
* Scikit-learn
* FastAPI
* Uvicorn
* Docker

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/janak8/risk-decision-system.git
cd risk-decision-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
python -m uvicorn api.app:app --reload
```

---

## 📡 API Endpoint

### POST `/predict`

### Example Request

```json
{
  "age": 35,
  "income": 60000,
  "loan_amount": 15000,
  "credit_score": 650,
  "years_employed": 5,
  "debt_to_income": 0.3
}
```

### Example Response

```json
{
  "risk": "LOW RISK",
  "probability": 0.23
}
```

---

## 🐳 Docker

### Build the image

```bash
docker build -t risk-api .
```

### Run the container

```bash
docker run -p 8000:8000 risk-api
```

---

## 📌 Future Improvements

* Add model explainability (SHAP / feature importance)
* Deploy on cloud (Render / AWS)
* Add authentication and logging
* Improve model performance with advanced algorithms

---

## 👨‍💻 Author

Janak Adhikari
