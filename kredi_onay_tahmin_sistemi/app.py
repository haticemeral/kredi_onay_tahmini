from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Model yükleme
with open("model/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["person_age"]),
            int(request.form["person_gender"]),
            int(request.form["person_education"]),
            float(request.form["person_income"]),
            float(request.form["person_emp_exp"]),
            int(request.form["person_home_ownership"]),
            float(request.form["loan_amnt"]),
            int(request.form["loan_intent"]),
            float(request.form["loan_int_rate"]),
            float(request.form["loan_percent_income"]),
            int(request.form["cb_person_cred_hist_length"]),
            float(request.form["credit_score"]),
            int(request.form["previous_loan_defaults_on_file"])
        ]

        prediction = model.predict([features])[0]

        result = "Kredi Onayı Verildi ✅" if prediction == 1 else "Kredi Onayı Reddedildi ❌"

        return render_template("result.html", result=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
