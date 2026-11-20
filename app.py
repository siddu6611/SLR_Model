from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open("SLR_Model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def default_site():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def prediction():

    if request.method == "POST":

        value = float(request.form["height"])

        value = np.array([[value]])

        result = model.predict(value)[0]

        return render_template("index.html", result=round(float(result), 2))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
