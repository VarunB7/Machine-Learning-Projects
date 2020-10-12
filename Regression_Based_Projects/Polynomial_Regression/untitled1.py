# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:17:16 2020

@author: hp
"""
print('1.add\n2.sub\n3.div\n4.mul\n')

print('choose op')
case=int(input())
print('enter 2 nos')
a=int(input())
b=int(input())
if case==1 :
    print('ans is',a+b)
elif case==2:
    print('ans is',a-b)
elif case==3:
    if  b!=0:
        print('ans is',a/b)
elif case==4:
    print('ans is',a*b)
    