# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 08:19:20 2023

"""

import nltk
nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
lemmatizer.lemmatize("programed")
lemmatizer.lemmatize("programs")

lemmatizer.lemmatize('battling')

lemmatizer.lemmatize('amazing')
#################################
#Chunking(Shallow Parsing) Identifying named entities
nltk.download("maxent_ne_chunker")
nltk.download("words")
#nltk.download('averaged_perceptron_tagger)
sentence4="we are learning NLP in python by SanjivaniAI b"
#first we will tokenize
from nltk import word_tokenize
words=word_tokenize(sentence4)
words=nltk.pos_tag(words)
i=nltk.ne_chunk(words,binary=True)
[ a for a in i if len(a)==1]
######################
#sentence tokenization
from nltk.tokenize import sent_tokenize
sent=sent_tokenize("we are learning NLP in Python. Delievered by SanjivaniAI. Do you know where it is located?It is in Kopargaon.")
sent
#
#########################
from nltk.wsd import lesk
sentence1="keep your savings in the bank"
print(lesk(word_tokenize(sentence1),'bank'))
#o/p:-Synset('savings_bank.n.02')
sentence2="It is so risky to drive over the banks of river"
print(lesk(word_tokenize(sentence2),'bank'))
#Synset('bank.v.07')
#############
#Synset('bank.v.07') a slope in the turn of road or track;
#the outside is higher than the inside in order to reduce the
#######
#'bank' as multiple meanings. if you want to find exact meaning 
#execute following code
#the definition for 'bank' can be seen here:
from nltk.corpus import wordnet as wn
for ss in wn.synsets('bank'): print(ss,ss.definition())


#
import re
sentence5="Sharad twitted ,wittnessing 70th republic day India from Rajpath,\new Delhi ,Memorizing performance by India Army! "
re.sub(r'([^\s\w]|_)+',' ',sentence5).split()
#extracting n-grams
#n- gram can be extracted using three techniques
#1.custom defined function
#2.nltk
#3.TextBlob
#################

#extracting n-grams using custom defined function

import re 
def n_gram_extractor(input_str,n):
    tokens=re.sub(r'([^\s\w]|_)+',' ',input_str).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])
        
n_gram_extractor("The cute little boy is playing with kitten",2)  #8-2+1
n_gram_extractor("The cute little boy is playing with kitten",3)  #8-3+1
######################

from nltk import ngrams
#extraction n-grams with nltk
list(ngrams("The cute little boy is playing with kitten".split(),2))
list(ngrams("The cute little boy is playing with kitten".split(),3))
####################

#pip install textblob
from textblob import TextBlob #IMP
blob=TextBlob('The cute little boy is playing with kitten.')
blob.ngrams(n=2)
blob.ngrams(n=3)
###################

#tokeniation using Keras
sentence5




