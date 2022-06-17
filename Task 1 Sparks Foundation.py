# By- ABHISHEK NAKWAL

## TASK 1- Prediction using Supervised Machine Learning

Dataset used:Student Scores
 



#import important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read the dataset as pandas dataframe
data=pd.read_csv("http://bit.ly/w-data")
data.head() #return top 5 rows

data.shape # return the shape i.e. number of rows and columns in data

data.info()

data.describe()

#### Visualize the data


sns.scatterplot(x=data['Hours'],y=data['Scores']);  #plot the data

sns.regplot(x=data['Hours'],y=data['Scores']);

#### Separate features and target

X=data[['Hours']]
y=data['Scores']

#### Train-Test Split

from sklearn.model_selection import train_test_split
 
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)

#### Model Building

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(train_X, train_y)  #train the model

pred_y = regressor.predict(val_X)  #prediction

pd.DataFrame({'Actual': val_y, 'Predicted':pred_y})  #view actual and predicted on test set side by side

# Actual vs Predicted distribution plot
sns.kdeplot(pred_y,label="Predicted", shade=True);

sns.kdeplot(data=val_y , label="Actual", shade=True);

print('Train accuracy: ',regressor.score(train_X, train_y),'\nTest accuracy: ',regressor.score(val_X,val_y))

h=[[9.25]]
s=regressor.predict(h)
print('A student who studies',h[0][0],'hours is estimated to score',s[0])

