# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 08:17:26 2023
"""
#tokenization using keras
import nltk
sentence5="Sharad twitted ,wittnessing 70th republic day India from Rajpath,\new Delhi ,Mesmorizing performance by Indian Army! "
sentence5
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(sentence5)
#######################

#Tokenization using TextBlob
from textblob import TextBlob
blob=TextBlob(sentence5)
blob.words

#######################
#tweet tokenizer
from nltk.tokenize import TweetTokenizer
tweet_tokenizer=TweetTokenizer()
tweet_tokenizer.tokenize(sentence5)
###############

#Multi-Word_Expression
from nltk.tokenize import MWETokenizer
sentence5
mwe_tokenizer=MWETokenizer([('republic','day')])
mwe_tokenizer.tokenize(sentence5.split())
mwe_tokenizer.tokenize(sentence5.replace('!',' ').split())
############################

#Regular Expression Tokenizer
from nltk.tokenize import RegexpTokenizer
reg_tokenizer=RegexpTokenizer('\w+|\$[\d\.]+|\S+')
reg_tokenizer.tokenize(sentence5)

'''

\w+|\$[\d\.]+|\S+
/
gm
1st Alternative \w+
\w matches any word character (equivalent to [a-zA-Z0-9_])
+ matches the previous token between one and unlimited times, as many times as possible, giving back as needed (greedy)
2nd Alternative \$[\d\.]+
\$ matches the character $ with index 3610 (2416 or 448) literally (case sensitive)
Match a single character present in the list below [\d\.]
+ matches the previous token between one and unlimited times, as many times as possible, giving back as needed (greedy)
\d matches a digit (equivalent to [0-9])
\. matches the character . with index 4610 (2E16 or 568) literally (case sensitive)
3rd Alternative \S+
\S matches any non-whitespace character (equivalent to [^\r\n\t\f\v ])
+ matches the previous token between one and unlimited times, as many times as possible, giving back as needed (greedy)
Global pattern flags 
g modifier: global. All matches (don't return after first match)
m modifier: multi line. Causes ^ and $ to match the begin/end of each line (not only begin/end of string)
'''

########################################
#white space tokenizer
from nltk.tokenize import WhitespaceTokenizer
wh_tokenizer=WhitespaceTokenizer()
wh_tokenizer.tokenize(sentence5)
#############################

#WordPuncTokenizer
from nltk.tokenize import WordPunctTokenizer
wp_tokenizer=WordPunctTokenizer()
wp_tokenizer.tokenize(sentence5)

##########################
#stemming
sentence6="I Love playing cricket. Cricket players practices hard in their inning"
from nltk.stem import RegexpStemmer
regex_stemmer=RegexpStemmer('ing$')
' '.join(regex_stemmer.stem(wd) for wd in sentence6.split())

###################################

##Porter Stemmer
sentence7="Before eating,it would be nice to sanitize your hands"
from nltk.stem.porter import PorterStemmer
ps_stemmer=PorterStemmer()
words=sentence7.split()
" ".join([ps_stemmer.stem(wd) for wd in words])

######################################
#Lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()
sentence8="The code executed today are far better than what we execute generally"
words=word_tokenize(sentence8)
" ".join([lemmatizer.lemmatize(word) for word in words])
##########################################

#singularize and pluralization
from textblob import TextBlob
sentence9=TextBlob("She sells seashells on the seashore")
words=sentence9.words
#we want to make word[2] i.e. seashells is singular form
sentence9.words[2].singularize()
#we want word 5 i.e. seashore in plural form
sentence9.words[5].pluralize()
############################

#language translation from spanish to English
from textblob import TextBlob
en_blob=TextBlob(u'muy bien')
en_blob.translate(from_lang='es',to='en')
#es:- spanish, en:- English

################################
#custom stopwords removal
from nltk import word_tokenize
sentence9="She sells seashells on the seashore"
custom_stop_word_list=['she','on','the','an','is']
words=word_tokenize(sentence9)
" ".join([word for word in words if word.lower()
          not in custom_stop_word_list])
#" ".join([word for word in words if word.lower()
#  not in custom_stop_word_list])
#select words which are not in defined list
#o/p:- 'sells seashells seashore'



