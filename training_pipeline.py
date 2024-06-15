import os
import sys
import pathlib
import mlflow

import numpy as np
import pandas as pd

from config import config
from processing.data_handling import load_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score

df = load_dataset(config.TRAIN_FILE)

def create_cohort(x):
    if x <=12:
        return '0-12 Months'
    elif x <=24:
        return '12-24 Months'
    elif x <=48:
        return '24-48 Months'
    else:
        return 'Over 48 Months'
    
df['Tenure_cohort'] = df['tenure'].apply(lambda x: create_cohort(x))

X = df.drop(['customerID','Churn'],axis=1)
y = df['Churn']

X = pd.get_dummies(df,drop_first=True)

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.1)

dt = DecisionTreeClassifier()
model_dt = dt.fit(X_train,y_train)

rf = RandomForestClassifier()
model_rf = rf.fit(X_train,y_train)

ada = AdaBoostClassifier()
model_ada = ada.fit(X_train,y_train)

gra = GradientBoostingClassifier()
model_gra = gra.fit(X_train,y_train)

def eval_metrics(actual,pred):
    accuracy = accuracy_score(actual,pred)
    f1 = f1_score(actual,pred)

    return (accuracy,f1)

def mlflow_logging(model,X,y,name):
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        pred = model.predict(X)

        (accuracy,f1) = eval_metrics(y,pred)
        
        mlflow.log_params(model.best_params_)

        mlflow.log_metric("Accuracy",accuracy)
        mlflow.log_metric("f1-score",f1)

        mlflow.sklearn.log_model(model,name)

        mlflow.end_run()

mlflow_logging(dt, X_test, y_test, "DecisionTreeClassifier")
mlflow_logging(ada, X_test, y_test, "AdaBoost")
mlflow_logging(rf, X_test, y_test, "RandomForestClassifier")
mlflow_logging(gra, X_test, y_test, "GradientBoosting")