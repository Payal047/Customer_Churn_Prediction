# 🚀 Customer Churn Prediction Dashboard

An end-to-end Machine Learning project that predicts customer churn and provides actionable insights using an interactive dashboard, REST API, database logging, and monitoring features.

---

## 📌 Project Overview

This project predicts whether a customer will churn based on key attributes such as tenure, monthly charges, contract type, and internet service.

It is designed as a **production-style ML system** with:

- Interactive dashboard (Dash)
- Backend API (FastAPI)
- Database logging (SQLite)
- Basic monitoring (drift detection)
- Deployment-ready structure

---

## 🎯 Key Features

- 📊 Interactive Dash-based dashboard
- 🤖 Machine Learning churn prediction model
- 📈 Probability visualization (Stay vs Churn)
- 💡 Rule-based explanation (reason for churn risk)
- 🌐 FastAPI REST endpoint for predictions
- 🗄️ SQLite database for prediction history
- 📉 Basic drift detection logic
- 🧾 Logging system for predictions

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Dash (Plotly)
- FastAPI
- SQLite
- Joblib
- Plotly Graph Objects

---

## 📂 Project Structure

```
customer-churn-project/
│
├── app.py                 # Dash dashboard
├── churn_model.pkl       # Trained ML model
├── app.log              # Logging file
├── churn_logs.db        # SQLite database
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Run Dashboard

```bash
python app.py
```

Dashboard will open at:

```
http://127.0.0.1:8050/
```

---

## 🌐 FastAPI Endpoint

Run API server:

```bash
uvicorn app:api --reload
```

### API Example:

```
GET /predict?tenure=12&monthly=50
```

### Response:

```json
{
  "prediction": 0,
  "probability": [0.54, 0.46]
}
```

---

## 🧠 Model Logic

The model uses customer features like:

- Tenure
- Monthly Charges
- Contract Type
- Internet Service

### Output:

- 0 → Customer will stay
- 1 → Customer will churn

---

## 💡 Business Insights

The system highlights risk factors such as:

- High monthly charges → higher churn risk
- Short-term contracts → unstable customers
- Low tenure → new customers likely to churn

---

## 🗄️ Database Logging

All predictions are stored in SQLite:

- tenure
- monthly charges
- prediction
- probability

Used for tracking and analysis.

---

## 📉 Monitoring

Basic drift detection is included to identify changes in input data distribution over time.

---

## 🐳 Docker (Optional)

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

---

## Dashboard 


http://127.0.0.1:8050/
---

## 🚀 Future Improvements

- Model retraining pipeline (Auto ML)
- User authentication system
- Cloud deployment (AWS / Render)
- Advanced explainability (SHAP/LIME)
- CI/CD pipeline integration

---


