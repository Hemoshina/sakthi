from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [float(x) for x in request.form.values()]
        features = np.array(inputs).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features).max() * 100

        return render_template('result.html', result=prediction, prob=round(probability, 2))
    except:
        return "Error: Please enter valid numerical inputs."

if __name__ == '__main__':
    app.run(debug=True)