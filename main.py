# FLASK_APP='main.py' flask run
import numpy as np
from flask import Flask, render_template, request

from model.tableaux import Tableaux
from model.priors import Priors
from model.report import Report
from optimize.regularizer import GaussianRegularizer
from optimize.learn import learn

app = Flask(__name__, static_folder="static")


@app.route("/", methods=["GET", "POST"])
def static_file():
    if request.method == "POST":
        tableaux = Tableaux.load(request.files["tableaux"])
        if "priors" in request.files:
            priors = Priors(filename=request.files["priors"])
        else:
            priors = Priors()

        reg = GaussianRegularizer(priors, tableaux.feature_names)
        beta0 = -5 * np.ones(len(tableaux.features))

        report = Report(tableaux, learn(beta0, tableaux, reg))
        return render_template("report.html", report=report)

    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="localhost", port=5050)
