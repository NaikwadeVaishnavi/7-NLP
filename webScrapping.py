# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:18:46 2023

@author: DELL5300 2IN -1
"""


##############################
#offline web scrapping
#pip install bs4
from bs4 import BeautifulSoup
soup = BeautifulSoup(open("C:/2-dataset/sample_doc.html"),'html.parser')

print(soup)
#it is going to show all the html contents extracted 

soup.text
#it will show all txt

soup.contents
#it is going to show all the html content extracted 

soup.find("address")
soup.find_all('address')
soup.find_all('q')
soup.find_all('b')
table = soup.find('table')

table

for row in table.find_all('tr'):
    columns = row.find_all('td')
    print(columns)
    
    
    
    
    
####################################
#online web scrapping

from bs4 import BeautifulSoup as bs
import requests

link = 'https://sajivanicoe.org.in/index.php/contact'

page = requests.get(link)

#Response [200] > it means connection is uccessfully established 

page.content 

#u will get all html source code but very crowdy txt
#let us apply html parser

soup = bs(page.content, 'html.parser')
soup 

#now the text is clean but not upto the expections
#now let us apply prettify method 

print(soup.prettify)

#the txt is clean and neat

list(soup.children)

#finding u want to extract contents from 
#first row

soup.find_all('p')[1].get_text()

#contents from second row 

soup.find_all('p')[1].get_text()

#finding text using class 

soup.find_all('div',class_='table')
