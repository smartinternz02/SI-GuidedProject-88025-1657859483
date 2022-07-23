import requests
import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import load_model

app=Flask(__name__)

model1=load_model(r'C:\Users\Hp\Plant Disease\training files\fruit pred.h5')
model=load_model(r'C:\Users\Hp\Plant Disease\training files\veg pred.h5')
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template('predict.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname('__file__')
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        img=image.load_img(file_path,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        plant=request.form['plant']
        print(plant)
        if(plant=="vegetable"):
            model=load_model(r'C:\Users\Hp\Plant Disease\training files\veg pred.h5')
            pred=np.argmax(model.predict(x),axis=1)
            index=['Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot']
          
            df=pd.read_excel('precautions - veg.xlsx')
            print(df.iloc[pred[0]]['caution'])
            text= index[pred[0]] + " Recommendation: " +str(df.iloc[pred[0]]['caution'])
        else:
            model2=load_model(r'C:\Users\Hp\Plant Disease\training files\fruit pred.h5')
            preds=np.argmax(model2.predict(x),axis=1)
            index=['Apple___Black_rot',
 'Apple___healthy',
'Corn_(maize)___Northern_Leaf_Blight',
'Corn_(maize)___healthy',
'Peach___Bacterial_spot',
'Peach___healthy']
            df=pd.read_excel('precautions - fruits.xlsx')
            print(df.iloc[preds[0]]['caution'])
            text= index[preds[0]] + " Recommendation: " +str(df.iloc[preds[0]]['caution'])
                      
        return text
        
 
if __name__=='__main__':
    app.run(debug=False)