import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request, jsonify

app = Flask(__name__)
current_path = os.getcwd()
pickle_path = os.path.join(current_path, "assets", "covid_predict.pkl")
classifier = pickle.load(open(pickle_path, "rb"))
data_path = os.path.join(current_path, "assets", "covid.csv")
df = pd.read_csv(data_path)

@app.route("/")
@app.route("/home")
def customer_details():
    return render_template(
        "index.html",
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    sex=request.form['sex']
    pneumonia=request.form['pneumonia']
    age=request.form['age']
    pregnancy=request.form['pregnancy']
    diabetes=request.form['diabetes']
    copd=request.form['copd']
    asthma=request.form['asthma']
    inmsupr=request.form['inmsupr']
    hypertension=request.form['hypertension']
    other_disease=request.form['other_disease']
    cardiovascular=request.form['cardiovascular']
    obesity=request.form['obesity']
    renal_chronic=request.form['renal_chronic']
    contact_other_covid=request.form['contact_other_covid']
    cols = [
        "sex",
        "pneumonia",
        "age",
        "pregnancy",
        "diabetes",
        "copd",
        "asthma",
        "inmsupr",
        "hypertension",
        "other_disease",
        "cardiovascular",
        "obesity",
        "renal_chronic",
        "contact_other_covid"
    ]
    test_data = pd.DataFrame(
        [
            [
        sex,
        pneumonia,
        age,
        pregnancy,
        diabetes,
        copd,
        asthma,
        inmsupr,
        hypertension,
        other_disease,
        cardiovascular,
        obesity,
        renal_chronic,
        contact_other_covid
            ]
        ],
        columns=cols,
    )
    pred = classifier.predict(test_data)
    if pred == 0:

        return render_template(
            "index.html", prediction_text="Covid-19 Negative")

    return render_template(
        "index.html", prediction_text="Covid-19 Positive")



if __name__ == "__main__":
    app.run(debug=True)