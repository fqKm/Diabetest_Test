from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
import pickle
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',probabilities=None, prediction=None, data=None)

@app.route('/post', methods=['POST'])
def predict():
    input_data = request.form
    df_input = pd.DataFrame([input_data])
    rfc = pickle.load(open('model.pkl', 'rb'))
    result = rfc.predict(df_input)
    probab = rfc.predict_proba(df_input)[0]
    probability_dictionary = {label: round(probability * 100, 2) for label, probability in zip(rfc.classes_, probab)}
    return render_template('index.html', prediction=result, probabilities=probability_dictionary, data=input_data)

if __name__ == '__main__':
    app.run(debug=False)