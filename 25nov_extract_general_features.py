# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 08:19:32 2023

"""
#practical natural language processing by orally publication

import nltk

#######################
#extracting general features from raw text
#number of words
#detect presence of wh word
#polarity
#subjectivity
#language identification
############################
#to identify the number of words
import pandas as pd
df=pd.DataFrame([['The vaccine for covid-19 will be announced on 1st August '],
                 ['Do you know how much expectations the world population is having from this research ?'],
                ['The rise of virus will come to an end on 31st July']])
df.columns=['text']
df

#############################
#now let us measure the number of words
from textblob import TextBlob
df['number_of_words']=df['text'].apply(lambda x:len(TextBlob(x).words))
df['number_of_words']

################################
#Detect presence of words wh
wh_words=set(['why','who','which','what','where','when','how'])
df['is_wh_words_present']=df['text'].apply(lambda x:True if len(set(TextBlob(str(x)).words).intersection(wh_words))>0 else False)
df['is_wh_words_present']

#####################################
#polarity of the sentence
df['polarity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']

sentence10="I like this example very much"
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10="This is fantastic example and I like it very much"
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10="This was helpful example but I would have prefer another example"
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10="This is my  personal opinion that it was helpful example but I would prefer another one"
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10="I like this example very much"
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10="I do not like this example very much"
pol=TextBlob(sentence10).sentiment.polarity
pol#o/p should be in negative

#this is used to analyze reviews 
#############################################

#subjective of the dataframe df and check whether there is personal opinion ...
df['subjectivity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.subjectivity)
df['subjectivity']

################################################
#to find lnguage of the sentence,this part of code will get http error
df['language']=df['text'].apply(lambda x:TextBlob(str(x)).detect_language())
df['language']=df['text'].apply(lambda x:TextBlob(str(x)).detect_language())
#run this code on google colab

#########################################
#bag of words
#




