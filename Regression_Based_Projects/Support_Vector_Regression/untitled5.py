# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 23:51:43 2020

@author: hp
"""

cardno = 4
cards = ['3695-7963-  5827-75',
'4143-4672-8798-2968-2968',
'6865---------------3965---------------1564-------------2918',
'6865396515642918']

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re



for j in range(cardno):
   
    x = re.findall("[a-zA-Z]", cards[j])
    arr = ['0000','1111','2222','3333','4444','5555','6666','7777','8888','9999','--']
    key = 0
    Leng = len(arr)
    k = 0
    if '-' in cards[j]:
        cardsp = cards[j].split("-")
        
        for i in range(len(cardsp)):
            if len(cardsp[i]) != 4:
                
                k = k+1
        cards[j] = re.sub("-","",cards[j])
    for i in range(Leng):
        if arr[i] in cards[j]:
            key = 1
    
 
    if len(x)>0  or  '_' in cards[j] or ' ' in  cards[j] or key != 0 or k != 0 or len(cards[j]) != 16:
         print("Invalid")
    else:
        if cards[j].startswith('4') or cards[j].startswith('5') or cards[j].startswith('6'):
            print("Valid")
           
            
            
           

        else:
           print("Invalid")
           

            





