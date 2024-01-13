# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:27:44 2023

"""

#################
#stemming
import nltk
stemmer=nltk.stem.PorterStemmer()
stemmer.stem("programming")
stemmer.stem("programmed")
stemmer.stem("Jumping")
stemmer.stem("Jumped")

#it gives root word
#such as for programming - program
#            programmed-program
#            Jumping-jump
#            Jumped-jump


################################
#Lematizer
#lematizer looks into dictionary words
nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
nltk.download('omw-1.4')
lemmatizer.lemmatize("programed")
lemmatizer.lemmatize("programs")

lemmatizer.lemmatize('battling')
lemmatizer.lemmatize('amazing')
#####################################

#Name Entity Recognition

#the code in google colab
