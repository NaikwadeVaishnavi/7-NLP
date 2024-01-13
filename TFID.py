# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:11:12 2023

@author: DELL5300 2IN -1
"""

###########################3


#how to use TFIDF

import pandas as pd
from  sklearn.feature_extraction.text import TfidfTransformer
from  sklearn.feature_extraction.text import CountVectorizer

corpus = ['The mouse had a tiny little mouse','The cat catch the mouse','The end of the mouse story']

#step initialise count vector

cv = CountVectorizer()

#to count the total no. of TF

word_count_vector = cv.fit_transform(corpus)
word_count_vector.shape

#now next step is to apply TF

tfidf_transformer = TfidfTransformer(smooth_idf= True, use_idf=True)

tfidf_transformer.fit(word_count_vector)

#this matrix is in the raw matrix form let us convert it in dataframe

df_idf = pd.DataFrame(tfidf_transformer.idf_, index= cv.get_feature_names_out(), columns= ["idf_weights"])

#sort ascending

df_idf.sort_values(by = ['idf_weights'])


###################


from  sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['Thar eating pizza, Loki eating pizza','Ironman ate pizza already',
          'Apple is announcing new iphone tommorow',
          'Tesla is announcing new model-3 tommorow',
          'Google is announcing new pixel-6 tommorow',
          'Microsoft is announcing new surface tommorow',
          'Amazon is announcing new eco-dot tommorow',
          'I am eating biryani and you are eating grapes']

#lets create the vectorizer and fit the corpos and transform them accordingly

v = TfidfVectorizer()
v.fit(corpus)
transform_output = v.transform(corpus)
#let us print vocabalary
print(v.vocabulary_)

#lets print the idf of each word 

all_feature_names = v.get_feature_names_out()

for word in all_feature_names:
    #lets get the index in the vocabalry
    
    indx = v.vocabulary_.get(word)
    # get the score 
    
    idf_score = v.idf_[indx]
    
    print(f"{word} : {idf_score}")
    

import pandas as pd
#read the data into  PANDAS dataframe

df = pd.read_csv("C:/2-dataset/Ecommerce_data.csv.xls")

print(df.shape)
df.head(5)

#check the distribution of labels

df['label'].value_counts()

#add the new column which gives a unique number to each of these label

df['label_num'] = df['label'].map({
    'Household':0, "Books":1,
    'Electronics':2,
    "Clothing & Accessories":3}
    )



#checking the result
df.head(5)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.Text,
    df.label_num,
    test_size=0.2,  # Corrected parameter name
    random_state=2022,
    stratify=df.label_num
)

print("Shape of X_train:",X_train.shape)

print("Shape of X_test:",X_test.shape)

print("Shape of y_train:",y_train.shape)

y_test.value_counts()

y_test.value_counts()

#################

#apply to classifier 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

#1. create a pipeline object 

clf = Pipeline([
    ('vectorizer_tfidf',TfidfVectorizer()),
    ("KNN",KNeighborsClassifier())
                  ])

#2. FIX WITH X_train and y_train 

clf.fit(X_train, y_train)


#3. fit the prediction for x train and store it i y pred

y_pred = clf.predict(X_test)

#4 PRINT the classification report 
print(classification_report(y_test, y_pred))
