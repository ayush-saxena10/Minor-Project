from flask import Flask, render_template, request
import requests
import jsonify
import numpy as np
from keras.models import load_model
import pickle

model = load_model('model.h5')
scaler_file = open('scaler.pickle', 'rb')
scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():    
    if request.method == 'POST':        
        post_age = float(request.form['age'])
        post_wb = float(request.form['wb'])
        post_fly_ash = float(request.form['fly_ash'])
        post_ab = float(request.form['ab'])

        prediction = model.predict(scaler.transform([[post_age, post_wb, post_fly_ash, post_ab]]))[0][0]
        prediction = prediction.round(decimals=2)
        prediction_text = f"Compressive Strength(MPa): {prediction}"
        return render_template("predict.html", text = prediction_text)
    else:
        return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)

