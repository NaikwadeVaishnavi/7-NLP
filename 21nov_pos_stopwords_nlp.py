# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:34:25 2023

"""
import nltk
from nltk import word_tokenize
words=word_tokenize("I am reading NLP Fundamentals")
print(words)
#parts of speech(PoS) tagging
nltk.download("averaged_perceptron_tagger")
nltk.pos_tag(words)
#it is going mention parts of speech
###################################
#stop words from NLTK library
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=stopwords.words('English')
#there are 179 stopwords in english language
#you can verify 179 stop words in variable explorer

print(stop_words)
sentence1="I am learning NLP:It is one of the most popular library in python"
#first we will tokenize the sentence
sentence_words=word_tokenize(sentence1)
print(sentence_words) 
#now let us filter the sentence using stop_words
sentence_no_stops=" ".join([words for words in sentence_words if words not in stop_words])
print(sentence_no_stops)
sentence1
#you can notice that am,is,of,the most,popular,in are missing 
###################################
#suppose we want to replace words in string
sentence2="I visited MY from IND on 14-02-19"
normalized_sentence=sentence2.replace("MY","Malasiya").replace("IND","India")
normalized_sentence=normalized_sentence.replace("-19","-2020")
print(normalized_sentence)
##########################
#suppose we want auto correction in the sentence
#pip install autocorrect
from autocorrect import Speller
#declare the function Speller defined for english
spell=Speller(lang='en')
spell("English") 
###########################

#suppose we want to correct whole sentence
sentence3="Ntural Langage processin deals withh the aart of extracting sentiiiments"
##let us first tokenize this sentence
sentence3=word_tokenize(sentence3)
corrected_sentence=" ".join([spell(word) for word in sentence3])
print(corrected_sentence)
