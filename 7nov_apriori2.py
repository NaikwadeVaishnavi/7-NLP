# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:11:55 2023


"""
#pip install mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
#here we are going to use transactinal data wherein size of each row is not
#we can not use pandas to load this unstructured data
#here function called open() is used
#create an empty list
groceries=[]
with open("C:/2-Dataset/groceries.csv.xls") as f:groceries=f.read()
#splitting the data into separate transactions using separator,it is comma 
#we can use new line character "\n"
groceries=groceries.split("\n")
#earlier groceries datastructure was in string format,now it will change to
#9836 , each item is comma separated
#our main aim is to calculate #A,#C
#we will have to separate out each item from each transaction
groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(","))
#split function will separate each item from each list,wherever it will find comma,it will split the item
#in order to generate associatio rules, you can directly use groceries_list
#now let us separate out each item from the groceries list
all_groceries_list={i for item in groceries_list for i in item}
#you will get all the item occured in all transactions
#we will get 43368 items in various transactions

#now let us count the frequency of each item 
#we will import collections package which has Counter function which will 
from collections import Counter 
item_frequencies=Counter(all_groceries_list)
#item_frequencies is basically dictionary having x[0] as key and x[1]=values
#we want to access values and sort based on the count that occured in it.
#it will show the count of each item purchased in every transaction
#now let us sort these frequencies in ascending order
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])
#when we execute  this , item frequencies will be in sorted form, in the form of
#in the form of tuple
#item name with count
#let us separate out items and their count
items=list(reversed([i[0] for i in item_frequencies]))
#this is list comprehension for each item in item frequencies access the key i,e. item
#here you will get items list
frequencies=list(reversed([i[1] for i in item_frequencies]))
#here you will get count of purchase of each item

#now let us plot bar graph of item frequencies
import matplotlib.pyplot as plt
#here we are taking  frequencies from zero to 11,you can try 0-15 or any other
plt.bar(height=frequencies[0:11],x=list(range(0,11))) 
plt.xticks(list(range(0,11)),items[0:11])
#plt.xticks,you can specify a rotation for the tick
#labels in degree or with keywords
plt.xlabel("items")
plt.ylabel("count")
plt.show()
import pandas as pd
#now let us try to establish association rule mining
#we have groceries list in list format,we need to convert it in dataframe format
groceries_series=pd.DataFrame(pd.Series(groceries_list))
#now we will get dataframe of size 9836X1 size,column comprises of multiple items
#we had extra row created,check the groceries_series, lastrow is empty,let us first delete it
groceries_series=groceries_series.iloc[:9835,:]
#we have taken rows from 0 to 9834 and column 0 to all
#groceries series has column having name 0,let us rename as transactions
groceries_series.columns=["Transactions"]
#now we will have to apply 1 hot encoding , before that in
#one column there are various items separated by ','
#',' let us separate it with '*'
x=groceries_series['Transactions'].str.join(sep='*')
#check the x in variable explorer which has * separator rather the ','
x=x.str.get_dummies(sep='*')
#you will get one hot encoded dataframe of size 9835X169
#this is our input data to apply to apriori algorithm, it will generate  !169 rules,min support value
#is 0.0075(it must be between 0 to 1)
#you can give any number but must be between 0 and 1
frequent_itemsets=apriori(x,min_support=0.0075,max_len=4,use_colnames='True')
#you will get suppport values for 1,2,3 and 4 max items
#let us sort these support values
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#support values will be sorted in descending order 
#even EDA was also have the same trend, in EDA there was count
#and here it is support value
#we will generate association rules,this association
#rule will calculate all the matrix
#of each and every combination
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)
#this generate association rules of size 1198X9 columns
#comprizes of antecedents,consequences
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)
