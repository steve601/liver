from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

def load_model():
    with open('liver.pkl','rb') as file:
        data = pickle.load(file)
    return data

objects = load_model()
model = objects['model']
preprocessor = objects['preprocessor']

@app.route('/')
def homepge():
    return render_template('liver.html')

@app.route('/classify',methods=['POST'])
def do_prediction():
    a = request.form.get('status')
    b = request.form.get('drug')
    c = request.form.get('age')
    d = request.form.get('sex')
    e = request.form.get('ascites')
    f = request.form.get('hepatomegaly')
    g = request.form.get('spiders')
    h = request.form.get('edema')
    i = request.form.get('bilirubin')
    j = request.form.get('cholesterol')
    k = request.form.get('albumin')
    l = request.form.get('copper')
    m = request.form.get('alk_phos')
    n = request.form.get('sgot')
    o = request.form.get('tryglicerides')
    p = request.form.get('platelets')
    q = request.form.get('prothrombin')
    
    columns = ['status', 'drug', 'age', 'sex', 'ascites', 'hepatomegaly','spiders', 'edema', 'bilirubin', 'cholesterol', 'albumin', 'copper','alk_phos', 'sgot', 'tryglicerides', 'platelets', 'prothrombin']
    x = pd.DataFrame([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q]],columns=columns)
    
    x = preprocessor.transform(x)
    pred = model.predict(x)
    
    if pred == 0:
        msg = 'The disease is at the first stage'
    elif pred == 1:
        msg = 'The disease is at the second stage'
    else:
        msg = 'The disease is at the third stage'
        
    return render_template('liver.html',text=msg)

if __name__ == '__main__':
    app.run(debug=True)