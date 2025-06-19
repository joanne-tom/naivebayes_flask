# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,ConfusionMatrixDisplay

dataset_path=r"C:\Users\Jonathan\Pictures\ml\naivebayes\Social_Network_Ads.csv"

# Initialize the Naive Bayes classifier
model = GaussianNB()

def loading_and_preprocessing(dataset_path):
    # Load the dataset
    data = pd.read_csv(dataset_path)
    print(data.head())
    data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})
    X = data[['Age', 'EstimatedSalary', 'Gender']]
    y = data['Purchased']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train,y_train):
    # Train the model on the training data
    model.fit(X_train, y_train)

def evaluate_model(X_test,y_test):
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {'accuracy': accuracy,'classification_report':report}
# con=confusion_matrix(y_test,y_pred)
# disp=ConfusionMatrixDisplay(confusion_matrix=con,display_labels=['male','female'])
# disp.plot()

def predict_model(sample):
    sample=np.array(sample).reshape(-1,3)
    purchase_prediction = model.predict(sample)
    prediction='Purchased' if purchase_prediction[0]==1 else 'Not Purchased'
    return {'Prediction':prediction}