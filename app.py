import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib
import pandas as pd
import plotly.graph_objs as go

# ------------------ LOAD MODEL ------------------
model = joblib.load("churn_model.pkl")

# ------------------ APP ------------------
app = dash.Dash(__name__)
server = app.server  # for deployment

# ------------------ STYLES ------------------
CARD_STYLE = {
    'padding': '20px',
    'borderRadius': '15px',
    'boxShadow': '0px 4px 15px rgba(0,0,0,0.1)',
    'backgroundColor': 'white'
}

TITLE_STYLE = {
    'textAlign': 'center',
    'padding': '20px',
    'color': '#1f2d3d',
    'fontWeight': 'bold'
}

# ------------------ LAYOUT ------------------
app.layout = html.Div([

    html.H1("🚀Customer Churn Prediction Dashboard", style=TITLE_STYLE),

    html.Div([

        # LEFT PANEL
        html.Div([

            html.H3("📥 Enter Customer Details", style={'marginBottom': '20px'}),

            html.Label("Tenure (months)"),
            dcc.Slider(
                id='tenure',
                min=0, max=72, step=1, value=12,
                marks={0: '0', 12: '12', 24: '24', 36: '36', 48: '48', 60: '60'}
            ),

            html.Br(),

            html.Label("Monthly Charges"),
            dcc.Input(
                id='monthly',
                type='number',
                value=50,
                style={'width': '100%', 'padding': '10px', 'borderRadius': '8px'}
            ),

            html.Br(), html.Br(),

            html.Label("Contract Type"),
            dcc.Dropdown(
                id='contract',
                options=[
                    {'label': 'Month-to-month', 'value': 'Month-to-month'},
                    {'label': 'One year', 'value': 'One year'},
                    {'label': 'Two year', 'value': 'Two year'}
                ],
                value='Month-to-month',
                clearable=False
            ),

            html.Br(),

            html.Label("Internet Service"),
            dcc.Dropdown(
                id='internet',
                options=[
                    {'label': 'DSL', 'value': 'DSL'},
                    {'label': 'Fiber optic', 'value': 'Fiber optic'},
                    {'label': 'No', 'value': 'No'}
                ],
                value='DSL',
                clearable=False
            ),

            html.Br(),

            html.Button(
                "Predict Customer Risk",
                id='btn',
                style={
                    'width': '100%',
                    'padding': '12px',
                    'borderRadius': '5px',
                    'border': 'none',
                    'backgroundColor': '#2563eb',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'cursor': 'pointer'
                }
            )

        ], style={**CARD_STYLE, 'width': '40%'}),

        # RIGHT PANEL
        html.Div([

            html.H3("📊 Prediction Result", style={'textAlign': 'center'}),

            html.Div(id='output', style={
                'fontSize': '22px',
                'textAlign': 'center',
                'marginTop': '20px',
                'fontWeight': 'bold'
            }),

            dcc.Graph(id='graph')

        ], style={**CARD_STYLE, 'width': '40%', 'marginLeft': '18px'})

    ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '15px', 'padding': '18px'})

])

# ------------------ CALLBACK ------------------
@app.callback(
    Output('output', 'children'),
    Output('graph', 'figure'),
    Input('btn', 'n_clicks'),
    State('tenure', 'value'),
    State('monthly', 'value'),
    State('contract', 'value'),
    State('internet', 'value')
)
def predict(n, tenure, monthly, contract, internet):

    if not n:
        return "Enter details and click predict", {}

    input_data = pd.DataFrame(columns=model.feature_names_in_)
    input_data.loc[0] = 0

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly

    # Encoding
    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1

    if internet == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    elif internet == "No":
        input_data["InternetService_No"] = 1

    input_data = input_data[model.feature_names_in_]

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    stay_prob, churn_prob = proba[0], proba[1]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Stay", "Churn"], y=[stay_prob, churn_prob]))
    fig.update_layout(
        title="Churn Probability",
        template="plotly_white"
    )

    reasons = []
    if monthly > 70:
        reasons.append("High charges")
    if tenure < 12:
        reasons.append("Low tenure")
    if contract == "Month-to-month":
        reasons.append("Short contract")

    reason_text = ", ".join(reasons) if reasons else "Low risk profile"

    if pred == 1:
        result = f"⚠️ High Risk: {churn_prob:.2f}"
    else:
        result = f"✅ Safe Customer: {stay_prob:.2f}"

    return result + " | " + reason_text, fig

# ------------------ RUN ------------------
if __name__ == '__main__':
    app.run(debug=True)

# ================== PRODUCTION ADD-ONS ==================

# 1. LOGGING
import logging
logging.basicConfig(filename="app.log", level=logging.INFO)

def log_prediction(input_data, prediction, prob):
    logging.info(f"INPUT: {input_data.to_dict()} | PRED: {prediction} | PROB: {prob}")

# 2. SQLITE DATABASE STORAGE
import sqlite3

conn = sqlite3.connect("churn_logs.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenure REAL,
    monthly REAL,
    prediction INTEGER,
    probability REAL
)
""")
conn.commit()

def save_to_db(tenure, monthly, prediction, probability):
    cursor.execute(
        "INSERT INTO predictions (tenure, monthly, prediction, probability) VALUES (?, ?, ?, ?)",
        (tenure, monthly, int(prediction), float(probability))
    )
    conn.commit()

# 3. SIMPLE DRIFT DETECTION (baseline mean shift)
def detect_drift(new_value, baseline_mean):
    return abs(new_value - baseline_mean) > (0.2 * baseline_mean)

# 4. FASTAPI BACKEND (API VERSION)
from fastapi import FastAPI
import uvicorn

api = FastAPI()

@api.get("/predict")
def api_predict(tenure: float, monthly: float):
    sample = pd.DataFrame([[tenure, monthly]], columns=["tenure", "MonthlyCharges"])
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0].tolist()
    return {"prediction": int(pred), "probability": prob}

# To run API separately:
# uvicorn filename:api --reload

# 5. DOCKER FILE (as reference)
# FROM python:3.10
# WORKDIR /app
# COPY . /app
# RUN pip install -r requirements.txt
# CMD ["python", "app.py"]
