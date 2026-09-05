from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('skmodel.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def default():
    return render_template('home.html')

@app.route('/predict',methods = ['post'])
def predict():
    data1 = float(request.form['val_a'])
    data2 = float(request.form['val_b'])
    data3 = float(request.form['val_c'])
    data4 = float(request.form['val_d'])

    input = np.array([[data1,data2,data3,data4]])

    prediction = model.predict(input)

    if prediction == 0:
        output = 'Iris-Setosa'
    if prediction == 1:
        output = 'Iris-Versicolor'
    if prediction == 2:
        output = 'Iris-Verginica'
    return render_template('home.html',prediction_value = 'Prediction is : {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
