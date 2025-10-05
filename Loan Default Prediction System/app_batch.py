from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "secretkey"

# ========================
# Define paths for models
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "model_columns.pkl")

# Load ML model, scaler, and columns
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
X_columns = joblib.load(COLUMNS_PATH)

# ========================
# Routes
# ========================

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/request_service', methods=['GET', 'POST'])
def request_service():
    if request.method == 'POST':
        try:
            # Extract borrower data from form
            borrower_data = {
                'age': int(request.form['age']),
                'gender': request.form['gender'],
                'education': request.form['education'],
                'employment_years': int(request.form['employment_years']),
                'loan_amount': float(request.form['loan_amount']),
                'term_months': int(request.form['term_months']),
                'interest_rate': float(request.form['interest_rate']),
                'installment': float(request.form['installment']),
                'purpose': request.form['purpose'],
                'annual_income': float(request.form['annual_income']),
                'dti': float(request.form['dti']),
                'credit_history_length': int(request.form['credit_history_length']),
                'open_accounts': int(request.form['open_accounts'])
            }

            # Convert to DataFrame
            df = pd.DataFrame([borrower_data])

            # One-hot encode categorical columns
            df = pd.get_dummies(df, columns=['gender','education','purpose'], drop_first=True)
            df = df.reindex(columns=X_columns, fill_value=0)

            # Scale numerical features
            df_scaled = scaler.transform(df)

            # Predict
            prediction = model.predict(df_scaled)[0]
            probability = model.predict_proba(df_scaled)[0][1]

            flash(f"Prediction: {'Default' if prediction==1 else 'No Default'}, Probability: {round(probability,3)}", 'success')
            return redirect(url_for('request_service'))

        except Exception as e:
            flash(f"Error: {str(e)}", 'danger')
            return redirect(url_for('request_service'))

    return render_template('request_service.html')


if __name__ == "__main__":
    app.run(debug=True)
