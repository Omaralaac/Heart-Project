import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model1.pkl','rb'))

@app.route('/')

def home():
    return render_template('index1.html')
@app.route('/predict', methods=['POST'])
def predict():

    int_features  = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0])

    if output==0:
        pred_pat = 'No Heart Disease'
    else :
        pred_pat = 'Has Heart Disease'

    return render_template('index1.html', prediction_text="The result of analysis {}".format(output), pred_pat=pred_pat)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

    

