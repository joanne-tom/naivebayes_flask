from flask import Flask,render_template,request,jsonify
import os
import numpy as np
from model.naive_bayes import loading_and_preprocessing,train_model,evaluate_model,predict_model

app=Flask(__name__)

dataset_path=r"data/Social_Network_Ads.csv"
X_train,X_test,y_train,y_test=loading_and_preprocessing(dataset_path)

train_model(X_train,y_train)

@app.route('/')
def index():
    return render_template('index.html')  # Render the home page

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/evaluate',methods=['GET'])
def evaluate():
    evaluation=evaluate_model(X_test,y_test)
    return jsonify({'Accuracy':evaluation['accuracy'],
                    'Classification report':evaluation['classification_report']})

@app.route('/predict',methods=['POST'])
def predict():
    print("POST request received")
    data=request.get_json()
    age=data.get('age')
    salary=data.get('salary')
    gender=data.get('gender')

    if gender is None or age is None or salary is None:
        return jsonify({'Error': 'Please provide age, salary, gender'})
    
    age_num=int(age)
    salary_num=int(salary)
    gender_num=int(gender)
    sample=np.array([[age_num,salary_num,gender_num]])
    prediction=predict_model(sample)
    return jsonify(prediction)

if __name__=='__main__':
    app.run(debug=True)
