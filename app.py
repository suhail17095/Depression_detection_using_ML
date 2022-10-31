import numpy as np
from flask import Flask, request, jsonify,  render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('depression_detection_model', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['get','post'])
def predict():
    features=[]
    for feature in request.form.values():
        try:
            features.append(float(feature))
        except:
            continue
    col=['age', 'gender', 'education', 'profession', 'maritial status',
       'residential place', 'living with', 'debt', 'smoke', 'drink', 'illness',
       'average sleep', 'insomia', 'work pressure', 'anxiety', 'depressed',
       'abused', 'cheat', 'threat', 'sucide', 'lost']
    array_features = pd.DataFrame(features,col)
    prediction = model.predict(array_features.T)
    prediction=prediction[0]
    if prediction == 1:
        return render_template("index.html",prediction_text=f"You are depressed please get some help",show="show")
    else:
        return render_template("index.html",prediction_text=f"You are not depressed",show="show")


if __name__=='__main__':
    app.run(debug=True)
