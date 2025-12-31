from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    # Show the form initially
    return render_template('home.html', prediction_text=None)

@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        if request.method == 'POST':
            # Get user input from HTML form
            data = CustomData(
                person_age=int(request.form['person_age']),
                person_gender=request.form['person_gender'],
                person_education=request.form['person_education'],
                person_income=float(request.form['person_income']),
                person_emp_exp=int(request.form['person_emp_exp']),
                person_home_ownership=request.form['person_home_ownership'],
                loan_amnt=float(request.form['loan_amnt']),
                loan_intent=request.form['loan_intent'],
                loan_int_rate=float(request.form['loan_int_rate']),
                loan_percent_income=float(request.form['loan_percent_income']),
                cb_person_cred_hist_length=int(request.form['cb_person_cred_hist_length']),
                credit_score=int(request.form['credit_score']),
                previous_loan_defaults_on_file=request.form['previous_loan_defaults_on_file']
            )

            input_df = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(input_df)

            # Convert 0/1 to Yes/No if needed
            result = "Approved" if prediction[0] == 1 else "Rejected"

            return render_template('home.html', prediction_text=result)

        else:
            # GET request → just show form
            return render_template('home.html', prediction_text=None)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
