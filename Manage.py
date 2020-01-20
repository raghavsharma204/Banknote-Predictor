from flask import Flask, render_template, request
import sqlite3 as sql
from sqlite3 import Error
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('Index.html')


@app.route('/Insert')
def Insert():
    return render_template('Insert.html')


@app.route("/predict", methods=["POST", "GET"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    print("final feet is ", final_features)
    prediction = model.predict(final_features.reshape(1, -1))
    print("prediction is ", prediction)
    if prediction == 1:
        authenticity = 'authentic'
    else:
        authenticity = 'fake'
    return render_template('Insert.html', msg='The banknote is {}'.format(authenticity))


if __name__ == '__main__':
    app.run(debug=True)
