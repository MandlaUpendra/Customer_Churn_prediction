# -*- coding: utf-8 -*-
"""Telecom_Customer_churn_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16mWBnHeHydcyuV8o3AByNjOs3N11J7tI

# Telco Customer Churn #

# This model helps in the anticipation of customer behavior to enhance customer retention, while also contributing to the formulation of a targeted customer retention strategy in Telecom Industry.#
"""

# Import all the necessary librarys.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier

df = pd.read_csv('/content/drive/MyDrive/Data/Data Science Masterclass data/Telco-Customer-Churn.csv')

"""**EDA(Exploratory Data Analysis)**"""

df.head()

df.info()
#In total there are 7032 datapoints available to feed our model.
#There are 2 float and int features present.
#Remaining 17 features are present in object datatype which requires transformation.

df.describe()
#Describe function only consider not-object features.

df.columns
#In total there are 21 columns present.

df.isnull().sum()
#We found that there are no Null values present.

sns.countplot(data=df,x='Churn');

"""*  churn" : No of customers that stopped using company product or services during ceratin period.
*   Here, we can clearly see in the countplot that the number of customers who remained with the company is relatively higher than those who dropped from the service.
"""

sns.violinplot(data=df,x='Churn',y='TotalCharges');
# The customers who churn in the begining stages paid way higher than who stayed for longer periods.

plt.figure(figsize=(12,8),dpi=200)
sns.boxplot(data=df,x='Contract',y='TotalCharges',hue='Churn');

"""

*   This boxplot totally make sense that, customers who took month-to-month contract are likely to exit in the beginning stages and there total charge is way too low. Likewise the one-year Contract customers have less total charges compared to two-year contract customers.

"""

df.columns

# Get_dummies is a powerful function that creats one-hot-encoding to all the object datatypes living out the float and int features.
# corr() tries to find the relation between all the features using pearson correlation, range is between -1 to 1.
corr_df = pd.get_dummies(df[['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines',
                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                            'PaymentMethod','InternetService','Churn']]).corr()

corr_df['Churn_Yes'].sort_values()[1:-1]

plt.figure(figsize=(10,4),dpi=200)
sns.barplot(x=corr_df['Churn_Yes'].sort_values()[1:-1].index,y=corr_df['Churn_Yes'].sort_values()[1:-1].values)
plt.title('Feature Correlation to Yes Churn')
plt.xlabel('Features')
plt.xticks(rotation=90);
# The below graph represent the correlation of all the features with Droped customers.
# It seeems the contract duration plays a crutial role in retation of customers, contract duration is 40% correlated with Churn.

plt.figure(figsize=(12,6),dpi=200)
sns.histplot(data=df,x='tenure',bins=60);

"""

1.   This Histgram of tenure revels the psychology of the customers; either they might exit at the beginning or they stay with the company once they've become accustomed to it."""

plt.figure(dpi=200)
sns.displot(data=df,x='tenure',col='Contract',row='Churn',bins=70);
#This displot adds further proof to the above Histgram, we company churn and different Contract tenures.

plt.figure(figsize=(12,6),dpi=200)
sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Churn',alpha=0.5);
#We already seen the same behaviour in the boxplot(churn vs Totalcharges), scatterplot shows different perception.

# we are trying to create a cohort for every tenure value present in the dataset and fing the percent of costumers who left the service with respect to each cohort.
no_churn =df.groupby(['Churn','tenure']).count().transpose()['No']
yes_churn = df.groupby(['Churn','tenure']).count().transpose()['Yes']

# This formula helps to find the percent of left cosutmers to total customers.
churn_rate = 100 * yes_churn / (no_churn + yes_churn)

churn_rate.transpose()['customerID']

# This is the visualization of above data.
plt.figure(figsize=(10,6),dpi=200)
plt.plot(churn_rate.transpose()['customerID'])
plt.xlabel('Tenure')
plt.ylabel('Churn Percentage');

"""*   we can clearly see that with tenure increase the customers are likely to stay with the company.

# **Feature Engineering.**
"""

# defining a functon that can segemt the customer according to there tenure.
def create_cohort(x):
  if x <=12:
    return '0-12 Months'
  elif x <=24:
    return '12-24 Months'
  elif x <= 48:
    return '24-48 Months'
  else:
    return 'Over 48 Months'

df['Tenure Cohort'] = df['tenure'].apply(lambda x: create_cohort(x))

"""### Here I am trying to segment the customers with respect to there tenure."""

df[['tenure','Tenure Cohort']]

plt.figure(figsize=(12,6))
sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Tenure Cohort');
# We segmented the totalcharges against the tenure cohort.

sns.countplot(data=df,x='Tenure Cohort',hue='Churn');

sns.displot(data=df,x='Tenure Cohort',col='Contract',hue='Churn');

plt.figure(figsize=(12,10),dpi=200)
sns.catplot(data=df,x='Tenure Cohort',col='Contract',hue='Churn',kind='count');

"""# **Model Implementation**"""

# I created a function that takes model as an argument and train with that model and print both classification report and confuison Matrix.
def run_model(model):
  model.fit(X_train,y_train)
  pred = model.predict(X_test)

  print(classification_report(y_test,pred))
  cm = confusion_matrix(y_test,pred)
  ConfusionMatrixDisplay(cm).plot()

# Dividing data into X(Indipendent data) and y(Dependent label)
X = df.drop(['customerID','Churn'],axis=1)
y = df['Churn']

# Creating dummy features so that the algoritham can underastand.
X = pd.get_dummies(df,drop_first=True)

# splitting the data keeping 10% of the data as test.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,f1_score

dt = DecisionTreeClassifier()

run_model(dt)

#Feeding the data to Decision Tree classifier with default parameters resulted in 100% accuracy.

dt.feature_importances_

from sklearn.tree import plot_tree

plt.figure(figsize=(12,8),dpi=150)
plot_tree(dt,filled=True,feature_names=X.columns);

rf = RandomForestClassifier()

run_model(rf)
#Feeding the data to Random Forest classifier with default parameters resulted in 100% precision and 96% Recall..

ada = AdaBoostClassifier()

run_model(ada)
#Feeding the data to Adaptive Boost classifier with default parameters resulted in 100% accuracy.

gra = GradientBoostingClassifier()

run_model(gra)
#Feeding the data to gradient Boost classifier with default parameters resulted in 100% accuracy.

"""# **Key Takeaways.**


*   We can see that most of the models are generating accuaracy of 100%, because the data we feed is very well structured and the volume of the data is very low with respect to models.
*   Even thought the accuracy might be 100%, we cannot declare that the future unseen data points also with get same results, the generalization problem might happend in the future data.
*   we can optimise the model if we recieve any further data points. And we haven't performed CV search either, so we can assume alot can go wrong in the genralization.

"""