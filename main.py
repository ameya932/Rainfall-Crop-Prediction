import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

app = Flask(__name__)

df = pd.read_csv("Crop_recommendation.csv")
le = LabelEncoder()
le.fit(df['label'])
cropmodel = tf.keras.models.load_model('Crop_model.h5')
rainmodel = tf.keras.models.load_model('Rainfall_model.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/crop', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('croppredict.html')
    else:
        N = request.form['N']
        P = request.form['P']
        K = request.form['K']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall = request.form['rainfall']

        model_input = [N, P, K, temperature, humidity, ph, rainfall]
        model_input = [float(i) for i in model_input]
        model_input = [model_input]

        prediction = cropmodel.predict(model_input)[0]
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
        
        return render_template('croppredict.html', crop=predicted_label)

@app.route('/predict/rain', methods = ['GET', 'POST'])
def predictrain():
    if request.method == 'GET':
        return render_template('rainpredict.html')
    else:
        m1 = request.form['m1']
        m2 = request.form['m2']
        m3 = request.form['m3']
        
        model_input = [float(m1), float(m2), float(m3)]
        model_input = np.asarray(model_input).reshape(1, 3, 1)
        prediction = rainmodel.predict(model_input)[0][0]
        return render_template('rainpredict.html', rainfall = prediction)
        
        




if __name__ == '__main__':
    app.run(debug=True)
