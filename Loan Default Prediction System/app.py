# app_swagger.py
from flask import Flask, request
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib

# ========================
# Load model, scaler, and columns
# ========================
model = joblib.load(r"C:\Users\admeni\OneDrive\Desktop\TCS AI Hackathon\Open Hack AI\logistic_regression_model.pkl")
scaler = joblib.load(r"C:\Users\admeni\OneDrive\Desktop\TCS AI Hackathon\Open Hack AI\scaler.pkl")
X_columns = joblib.load(r"C:\Users\admeni\OneDrive\Desktop\TCS AI Hackathon\Open Hack AI\model_columns.pkl")

# ========================
# Initialize Flask + RESTX
# ========================
app = Flask(__name__)
api = Api(app, version='1.0', title='Loan Default Prediction API',
          description='Predict if a borrower will default on a loan')

ns = api.namespace('predict', description='Prediction operations')

# ========================
# Define input model for Swagger
# ========================
borrower_model = api.model('Borrower', {
    'age': fields.Integer(required=True, description='Age of borrower'),
    'gender': fields.String(required=True, description='Gender: Male/Female'),
    'education': fields.String(required=True, description='Education level'),
    'employment_years': fields.Integer(required=True, description='Years employed'),
    'loan_amount': fields.Float(required=True, description='Loan amount requested'),
    'term_months': fields.Integer(required=True, description='Loan term in months'),
    'interest_rate': fields.Float(required=True, description='Interest rate (%)'),
    'installment': fields.Float(required=True, description='Monthly installment'),
    'purpose': fields.String(required=True, description='Loan purpose'),
    'annual_income': fields.Float(required=True, description='Annual income'),
    'dti': fields.Float(required=True, description='Debt-to-Income ratio'),
    'credit_history_length': fields.Integer(required=True, description='Credit history length in years'),
    'open_accounts': fields.Integer(required=True, description='Number of open credit accounts')
})

# ========================
# Prediction endpoint
# ========================
@ns.route('/')
class LoanDefaultPredictor(Resource):
    @ns.expect(borrower_model)
    def post(self):
        try:
            borrower_data = request.json
            
            # Convert to DataFrame
            new_data = pd.DataFrame([borrower_data])

            # One-hot encode categorical columns
            new_data = pd.get_dummies(new_data, columns=['gender','education','purpose'], drop_first=True)

            # Align with training columns
            new_data = new_data.reindex(columns=X_columns, fill_value=0)

            # Scale numerical features
            new_data_scaled = scaler.transform(new_data)

            # Predict
            prediction = model.predict(new_data_scaled)[0]
            probability = model.predict_proba(new_data_scaled)[0][1]

            return {
                'prediction': 'Default' if prediction == 1 else 'No Default',
                'default_probability': round(probability, 3)
            }

        except Exception as e:
            return {'error': str(e)}, 400

# ========================
# Home route
# ========================
@app.route('/')
def home():
    return "Loan Default Prediction API with Swagger UI is running!"

# ========================
# Run the app
# ========================
if __name__ == '__main__':
    app.run(debug=True)
