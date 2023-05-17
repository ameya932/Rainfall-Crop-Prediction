import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, redirect
import requests
from bs4 import BeautifulSoup

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
def predict_crop():
    if request.method == 'GET':
        if request.args.get('rainfall', None):
            return render_template('croppredict.html', rainfall = request.args.get('rainfall', None))
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

        crop_info, crop_image_url, irrigation_link = fetch_crop_details(predicted_label)

        return render_template('croppredict.html', crop=predicted_label, crop_info=crop_info,
                               crop_image_url=crop_image_url, irrigation_link=irrigation_link)


def fetch_crop_details(crop_name):
    crop_info = ""
    crop_image_url = ""
    irrigation_link = ""

    url = f'https://en.wikipedia.org/wiki/{crop_name}'
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        crop_info_element = soup.find('div', {'id': 'mw-content-text'})
        if crop_info_element:
            paragraphs = crop_info_element.find_all('p')
            crop_info = ' '.join([p.text for p in paragraphs])

        image_element = soup.find('img', {'class': 'thumbimage'})
        if image_element:
            crop_image_url = f"https:{image_element['src']}"

        irrigation_link = f'https://wikifarmer.com/{crop_name}'

    if len(crop_info) > 1000:
        crop_info = crop_info[:1000] + "..."

    return crop_info, crop_image_url, irrigation_link


@app.route('/predict/rain/custom', methods=['POST'])
def predict_rain_custom():
    if request.method == 'POST':
        m1 = request.form['m1']
        m2 = request.form['m2']
        m3 = request.form['m3']

        model_input = [float(m1), float(m2), float(m3)]
        model_input = np.asarray(model_input).reshape(1, 3, 1)
        print(model_input)
        prediction = round(rainmodel.predict(model_input)[0][0], 2)
        gotocrop = False
        try:
            y = request.form['gotocrop']
            gotocrop = True
        except:
            pass
        if gotocrop:
            return redirect('/predict/crop?rainfall='+str(prediction))
        return render_template('rainpredict.html', rainfall_custom="Predicted rainfall: "+str(prediction)+"mm")

    return render_template('rainpredict.html')

@app.route('/predict/rain', methods=['GET', 'POST'])
def predict_rain():
    if request.method == 'GET':
        return render_template('rainpredict.html')
    else:
        state = request.form['state']
        month = request.form['month']

        month_mapping = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

        try:
            month_value = month_mapping[month.upper()]
        except KeyError:
            return render_template('rainpredict.html', error_message='Invalid month entered.')

        custom_data = np.array([[month_value] * 3])
        model_input = np.expand_dims(custom_data, axis=2)
        print(model_input)
        prediction = rainmodel.predict(model_input)
        gotocrop = False
        try:
            y = request.form['gotocrop']
            gotocrop = True
        except:
            pass
        if gotocrop:
            return redirect('/predict/crop?rainfall='+str(prediction[0][0]))
        return render_template('rainpredict.html', rainfall="Predicted rainfall: "+str(prediction[0][0])+"mm")
    


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)




# if __name__ == '__main__':
#     app.run(debug=True)

