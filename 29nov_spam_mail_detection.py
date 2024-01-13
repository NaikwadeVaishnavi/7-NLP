# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:50:16 2023

@author: rohit
"""

import pandas as pd 
import numpy as np
#read the csv
df=pd.read_csv("C:/2-dataset/spam.csv.xls")
#check first 10 records
df.head()
#Total number of spam and ham
df.Category.value_counts()
#create  one more column comprises 0 to 1
#name of column is spam

df['spam']=df['Category'].apply(lambda x: 1 if x =='spam' else 0)
df.shape

###############
#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.Message,df.spam,test_size=0.2)

#let us  check the shape of x_train and x_test
x_train.shape
x_test.shape

#check the type of x_train y_train
type(x_train)
type(y_train)

########################
#create bag of words representation using countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
x_train_cv=v.fit_transform(x_train.values)
x_train_cv

#after creation of BoW let us check the shape
x_train_cv.shape
#######################

#train the naive bayes model
from sklearn.naive_bayes import MultinomialNB
#initialize the model
model=MultinomialNB()
#train the model
model.fit(x_train_cv,y_train)

###############################
#create bag of words representation using CountVectorizer
#of x_test
x_test_cv=v.transform(x_test)
#########################

#evaluate the performance
from sklearn.metrics import classification_report
y_pred=model.predict(x_test_cv)
print(classification_report(y_test,y_pred))

