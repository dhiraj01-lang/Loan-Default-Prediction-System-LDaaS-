import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ========================
# Load trained model & scaler
# ========================
model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")   # save your scaler separately during training

# ========================
# Function to predict loan default
# ========================
def predict_default(borrower_data: dict):
    """
    borrower_data: dict of borrower info
    Example:
    {
        'age': 35,
        'gender': 'Male',
        'education': 'Bachelors',
        'employment_years': 5,
        'loan_amount': 20000,
        'term_months': 60,
        'interest_rate': 12,
        'installment': 450,
        'purpose': 'Car',
        'annual_income': 55000,
        'dti': 0.25,
        'credit_history_length': 6,
        'open_accounts': 5
    }
    """
    # Convert dict to DataFrame
    new_data = pd.DataFrame([borrower_data])

    # One-hot encode like training set
    new_data = pd.get_dummies(new_data, columns=['gender','education','purpose'], drop_first=True)

    # Align with training columns
    X_columns = joblib.load("model_columns.pkl")   # save training feature columns earlier
    new_data = new_data.reindex(columns=X_columns, fill_value=0)

    # Scale numerical features
    new_data_scaled = scaler.transform(new_data)

    # Predict
    prediction = model.predict(new_data_scaled)[0]
    probability = model.predict_proba(new_data_scaled)[0][1]

    return prediction, probability


# ========================
# Example Run
# ========================
if __name__ == "__main__":
    borrower = {
        'age': 42,
        'gender': 'Female',
        'education': 'Masters',
        'employment_years': 10,
        'loan_amount': 15000,
        'term_months': 36,
        'interest_rate': 9,
        'installment': 400,
        'purpose': 'Education',
        'annual_income': 72000,
        'dti': 0.22,
        'credit_history_length': 12,
        'open_accounts': 6
    }

    pred, prob = predict_default(borrower)
    print("Prediction:", "Default" if pred==1 else "No Default")
    print("Default Probability:", round(prob,3))
