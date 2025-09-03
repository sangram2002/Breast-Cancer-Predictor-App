

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

def create_model(data):
    x = data.drop(['diagnosis'],axis=1)
    y = data['diagnosis']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    model = LogisticRegression()
    model.fit(x,y)
    return model,scaler


    # split into train and test
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    return model,scaler


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def create_model(data):
    x = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # Predictions
    y_pred = model.predict(x_test)
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    # Classification Report
    print("Classification Report:\n", classification_report(y_test, y_pred))
    # Confusion Matrix
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return model, scaler


def main():
    data = get_clean_data()
    print(data.head())
    model,scaler = create_model(data)

    # Save the model
    with open ('model.pkl','wb') as f:
        pickle.dump(model,f)
    with open ('scaler.pkl','wb') as f:
        pickle.dump(scaler,f)




def get_clean_data():
    data = pd.read_csv('D:\Data Science\Extra Projects\Breast-Cancer-Prediction\data.csv')
    print(data.head())
    data=data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    return data



if __name__== '__main__':
    main()    