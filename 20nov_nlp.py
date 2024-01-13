# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:26:31 2023

"""

####################Text Mining#############
sentence="we are learning TextMining from Sanjivani AI"
###if we want to know position of learning
sentence.index("learning")
##########it will showlearning is at position 7
#This is going to show character position from 0 including
#############################
# we want to know position TextMining word

sentence.split().index("TextMining")
#it will split the words in list and count the position 
###if you want to see the list select sentence.split() and
#it will show at 3
####################
#suppose we want print any word in reverse order
sentence.split()[2][::-1]
#[start:end end:-1(start)] will start from -1,-2,-3 till the end
##learning will be printed as gninrael
#############################

#suppose want to print first and last word of sentence
words=sentence.split()
first_word=words[0]
first_word
last_word=words[-1]
last_word
###now we want to concatenate the first and last word

concat_word=first_word+" "+last_word
concat_word
################################

#we want to print even words from sentences
[words[i] for i in range(len(words)) if i%2==0]
##words having odd length will not be printed
#########################
#we want to print odd words from sentences
[words[i] for i in range(len(words)) if i%2!=0]
##################################

###################
sentence
#now we want to display only AI
sentence[-3:]
#it will start from -3,-2,-1 i,e. AI
##################
#suppose we want to display entire sentence in reverse order
sentence[::-1]
#'IA inavijnaS morf gniniMtxeT gninrael era ew'
#here whole sentence is get reversed
###################

#suppose we want to select each word and print in reversed order
words
print(" ".join(word[::-1] for word in words))
#'IA inavijnaS morf gniniMtxeT gninrael era ew'
#here only letters get reversed

######################
#Tokenization
import nltk
nltk.download('punkt')
from nltk import word_tokenize
words=word_tokenize("I am reading NLP Fundamentals")
print(words)
###################


#21 Nov. 2023
#parts of speech(PoS) tagging
nltk.download("averaged_perceptron_tagger")
nltk.pos_tag(words)
#it is going mention parts of speech
###################################
#stop words from NLTK library
from nltk.corpus import stopwords
stop_words=stopwords.words("English")
#there are 179 stopwords in english language
