# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:07:19 2023

@author: DELL5300 2IN -1
"""

from bs4 import BeautifulSoup as bs
import requests
link='https://www.flipkart.com/canon-eos-m50-mark-ii-mirrorless-camera-ef-m15-45mm-stm-lens/p/itm7a4f536cb1255?pid=DLLGFY7XYG8YFMQT&lid=LSTDLLGFY7XYG8YFMQTSG43XC&marketplace=FLIPKART&store=jek%2Fp31%2Ftrv&srno=b_1_1&otracker=browse&fm=organic&iid=5423067e-4956-42c5-96fc-7859ea21a79c.DLLGFY7XYG8YFMQT.SEARCH&ppt=hp&ppn=homepage&ssid=4k8668ei0w0000001701745668167'
page=requests.get(link)
page                           # <Response [200]> successful
page.content
soup=bs(page.content,'html.parser')
print(soup.prettify())
title=soup.find_all('p',class_="_2-NBzT")
title
review_title=[]
for i in range (0,len(title)):
    review_title.append(title[i].get_text())
review_title
len(review_title)

#we got 10 reviews title 
## now let us scrap rating
