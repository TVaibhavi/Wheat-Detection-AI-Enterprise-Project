# -*- coding: utf-8 -*-



import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
from collections.abc import Mapping

scaler = MinMaxScaler()

app=Flask(__name__)
pickle_in = open("selected_model.pkl","rb")
selected_model = pickle.load(pickle_in)
min_data = [10.59, 12.41,	0.8081,	4.899,	2.63,	0.7651,	4.519]
max_data = [21.18,	17.25,	0.9183,	6.675,	4.033,	8.315,	6.55]
cols = ['Area', 'Perimeter', 'Compactness', 'Kernel.Length', 'Kernel.Width',
       'Asymmetry.Coeff', 'Kernel.Groove']
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    final_features = [float(x) for x in request.form.values()]
    print(final_features)
    print(min_data)
    print(max_data)

    min_df = pd.DataFrame(min_data).transpose()
    min_df.columns = cols
    print(min_df)

    max_df = pd.DataFrame(max_data).transpose()
    max_df.columns = cols
    print(max_df)

    final_df = pd.DataFrame(final_features).transpose()
    final_df.columns = cols
    print(final_df)

    test_df = final_df.append([max_df,min_df],ignore_index = True)
    test_norm_df = pd.DataFrame(scaler.fit_transform(test_df))
    test_norm_df.columns = test_df.columns
    print(test_norm_df)
    prediction = selected_model.predict(test_norm_df.head(1))

    print(prediction)
    output= 'not available'
    if prediction[0]==1:
      output = 'Kama'
    elif prediction[0]==2:
      output='Rosa'
                 
    elif prediction[0]==3:
      output='Canadian'
            
    print(output)
    
    return render_template('index.html', prediction_text='The wheat seed belongs to type {}'.format(output))
    
    
    


if __name__=='__main__':
    app.run()