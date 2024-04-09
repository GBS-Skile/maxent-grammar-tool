# FLASK_APP='src/app.py' flask run
import numpy as np
from flask import Flask, request

from model.tableaux import Tableaux
from model.priors import Priors
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

        return str(learn(beta0, tableaux, reg))

    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="localhost", port=5050)
