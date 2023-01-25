from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('model.pkl','rb'))
df=pd.read_csv('clean.csv')

@app.route('/',methods=['GET','POST'])
def index():
    age=sorted(df['age'].unique())
    sex=sorted(df['sex'].unique())
    bmi=sorted(df['bmi'].unique())
    children=sorted(df['children'].unique())
    smoker=sorted(df['smoker'].unique())
    region=sorted(df['region'].unique())
    
    
    return render_template('index.html',age=age, sex=sex, bmi=bmi,children=children,smoker=smoker,region=region)

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    age=request.form.get('age')

    sex=request.form.get('sex')
    bmi=request.form.get('bmi')
    children=request.form.get('children')
    smoker=request.form.get('smoker')
    region=request.form.get('region')

    prediction=model.predict(pd.DataFrame(columns=['age', 'sex', 'bmi', 'children', 'smoker','region'],
                              data=np.array([age,sex,bmi,children,smoker,region]).reshape(1, 6)))
    print(prediction)

    return str(np.round(prediction[0],2))




if __name__=='__main__':
    app.run()