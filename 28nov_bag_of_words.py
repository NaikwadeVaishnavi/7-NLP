# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:15:45 2023

@author: rohit
"""

import nltk
import pandas as pd
#corpus-set of sentences is called corpus.
#document-single sentence is called document.
from sklearn.feature_extraction.text import CountVectorizer
corpus=['At least seven idian pharma companies are working to develop vaccine against the corona virus.','The deadly virus that has already infected more than 14 million globally','Bharat Biotech is the domastic pharma firm working on corona virus vaccine in India']
bag_of_words_model=CountVectorizer()
print(bag_of_words_model.fit_transform(corpus).todense())
bag_of_words_df=pd.DataFrame(bag_of_words_model.fit_transform(corpus).todense())
#This will create dataframe
bag_of_words_df.columns=sorted(bag_of_words_model.vocabulary_)
bag_of_words_df.head()
####################################
bag_of_words_model_small=CountVectorizer(max_features=5)
bag_of_words_df_small=pd.DataFrame(bag_of_words_model_small.fit_transform(corpus).todense())
bag_of_words_df_small.columns=sorted(bag_of_words_model_small.vocabulary_)
bag_of_words_df_small.head()

######################################

