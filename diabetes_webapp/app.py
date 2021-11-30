import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    prediction_text = 'Hello'
    if prediction==1:
        prediction_text = "High Chance of Diabetes"
    else:
        prediction_text = "Low Chances of Diabetes"
    print(prediction_text)
    return render_template('index.html', prediction_text=prediction_text)


if __name__ == '__main__':
    app.run(debug=True)