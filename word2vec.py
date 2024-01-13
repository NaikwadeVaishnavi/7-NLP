# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:35:50 2023

@author: DELL5300 2IN -1
"""

#pip install gensim
#pip install python-Levenshtein

import gensim
import pandas as pd

df = pd.read_json("C:/2-dataset/Cell_Phones_and_Accessories_5.json",lines=True)

df

df.shape

#simple preprocessing and Tokenization

review_text = df.reviewText.apply(gensim.utils.simple_preprocess)

review_text

#let us check first word of each review

review_text.loc[0]

#let us check first row of dataframe 
    
df.reviewText.loc[0]

#training the Word2Vec Model

model = gensim.models.Word2Vec(window= 10 , min_count=2, workers=4,)

'''
where window is how many words you are going to consider as 
slicing window u can choose any count min_count-there must 2 words in each 
sentence worker : no. of threads
'''


#build vocabuulary 

model.build_vocab(review_text, progress_per=1000)

#progress_per : after 1000 words it shows progress
#train the Word2Vec Model
#it will take time have  patience 

model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)

#save the model

model.save("C:/7-NLP./word2vec-amazon-cell-accessories-review-short.model")

#finding similar word and similarity between words

model.wv.most_similar("bad")
model.wv.similarity(w1= 'cheap', w2='inexpensive')
model.wv.similarity(w1= 'great', w2= 'good')
