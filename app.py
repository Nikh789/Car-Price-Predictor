from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
cars=pd.read_csv("Car Data123.csv")

@app.route('/')
def index():
    companies = sorted(cars['company'].unique())
    ownerused = sorted(cars['owner'].unique())
    yearold = sorted(cars['year'].unique(), reverse=True)
    fuel_type = cars['fuel'].unique()
    return render_template('index.html', companies=companies, ownerused=ownerused, yearold=yearold, fuel_type=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company=request.form.get('company')
    owner=request.form.get('owner')
    year=int(request.form.get('year'))
    fuel=request.form.get('fuel')
    km_driven=int(request.form.get('km_driven'))

    prediction = model.predict(pd.DataFrame([[company, owner, year, fuel, km_driven]], columns=['companies', 'ownerused', 'yearold', 'fuel_type', 'km_driven']))

    return str(np.round(prediction[0], 2))

if __name__=="__main__":
    app.run(debug=True)