# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

t = pd.read_csv('C:\\Users\\Shriyash Shende\\Desktop\\titanic\\train.csv')
#EDA

t.isnull()
t.info()
sns.heatmap(t.isnull(), cbar=False)
t.columns

t['Age'].isnull().sum()
t['Cabin'].isnull().sum()
t['Embarked'].isnull().sum()
t['Age'].mean()
median = t['Age'].median()
t['Age'].fillna(median, inplace = True)
sns.heatmap(t.isnull(), cbar=False)
mode = t['Embarked'].mode()
t['Embarked'].fillna('S', inplace = True)
sns.heatmap(t.isnull(), cbar=False)
t['Embarked'].isnull().sum()

t.columns

t1 = t.drop(['PassengerId', 'Name', 'Cabin'], axis = 1)
t1.corr()

###Altercation Ticket Variable

for i in t1['Ticket']:
    j = str(i)
    if(len(j) == 6 or len(j) == 7):
        if(j[0:3] == '111' or j[0:3] == '112' or j[0:3] == '113' or j[0:3] == '114' or j[0:3] == '115' or j[0:3] == '116'):
            t1['Ticket'].replace(to_replace = j, value ="T", inplace = True)
        
        elif(j[0:2] == 'CA' or j[0:2] == 'C '):
            t1['Ticket'].replace(to_replace = j, value ="CA", inplace = True)
        elif(j[0:2] == 'SC'):
            t1['Ticket'].replace(to_replace = j, value ="A", inplace = True)
        elif(j[0:2] == 'PP'):
            t1['Ticket'].replace(to_replace = j, value ="P", inplace = True)
        else:
            t1['Ticket'].replace(to_replace = j, value ="S", inplace = True)
    elif(len(j) == 4):
        if(j[0] == '9' or j[0] == '3' or j[0] == '4'):
            t1['Ticket'].replace(to_replace = j, value ="P", inplace = True)
        else:
            t1['Ticket'].replace(to_replace = j, value ="CA", inplace = True)

g = list(range(11600,12000,1))
p = list(range(12000,20000,1))
c = list(range(20000,30000,1))
a = list(range(30000,70000,1))             
for i in t1['Ticket']:
    j = str(i)
    if(len(j) == 5):
        if int(j) in g:
            t1['Ticket'].replace(to_replace = j, value ="G", inplace = True)
        elif(int(j) in p):
            t1['Ticket'].replace(to_replace = j, value ="PC", inplace = True)
        elif(int(j) in c):
            t1['Ticket'].replace(to_replace = j, value ="CA", inplace = True)
        elif(int(j) in a):
            t1['Ticket'].replace(to_replace = j, value ="A", inplace = True)

for i in t1['Ticket']:
    j = str(i)
    if(len(j) > 2):
        if(j[0]=='C' or j[0]=='W'):
            t1['Ticket'].replace(to_replace = j, value ="CA", inplace = True)
        elif(j[0]== 'A'):
            t1['Ticket'].replace(to_replace = j, value ="A", inplace = True)
        elif(j[0:2] == 'SC'):
            t1['Ticket'].replace(to_replace = j, value ="A", inplace = True)
        else:
            print(j)
          
for i in t1['Ticket']:
    j = str(i)
    if(len(j) > 3):
        if(j[0:3] == 'S.C'):
            t1['Ticket'].replace(to_replace = j, value ="A", inplace = True)
        if(j[0:2] == 'PC'):
            t1['Ticket'].replace(to_replace = j, value ="PC", inplace = True)
        if(j[0:4] == 'STON' or j[0:5] == 'SOTON'):
            t1['Ticket'].replace(to_replace = j, value ="S", inplace = True)
        if(j[0] == 'P' or j[0:2] == 'SW' or j[0:3] == 'S.W'):
            t1['Ticket'].replace(to_replace = j, value ="P", inplace = True)
        if(j[0] == 'F'):
            t1['Ticket'].replace(to_replace = j, value ="F", inplace = True)
        else:
            t1['Ticket'].replace(to_replace = j, value ="SOC", inplace = True)
           
for i in t1['Ticket']:
    j = str(i)
    if j.isnumeric():
        t1['Ticket'].replace(to_replace = j, value ="A", inplace = True)
  
#####################################3
        
sns.countplot(t1['Ticket'])

t2 = pd.get_dummies(t1, drop_first = True)       
X = t2.drop(['Survived'], axis = 1)
Y = t2['Survived']      

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.20, random_state=0)   

clf = LogisticRegression()
clf.fit(X_train, Y_train)            
pred = clf.predict(X_test)       

from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test, pred)        

per = cn[0,0] + cn[1,1]
p = per + cn[0,1] + cn[1,0]
(per / p) * 100


test = pd.read_csv('C:\\Users\\Shriyash Shende\\Desktop\\titanic\\test.csv')
test['Age'].fillna(median, inplace = True)
test['Embarked'].fillna('S', inplace = True)      
test1 = test.drop(['PassengerId', 'Name', 'Cabin'], axis = 1)        
for i in test1['Ticket']:
    j = str(i)
    if(len(j) == 6 or len(j) == 7):
        if(j[0:3] == '111' or j[0:3] == '112' or j[0:3] == '113' or j[0:3] == '114' or j[0:3] == '115' or j[0:3] == '116'):
            test1['Ticket'].replace(to_replace = j, value ="T", inplace = True)
        
        elif(j[0:2] == 'CA' or j[0:2] == 'C '):
            test1['Ticket'].replace(to_replace = j, value ="CA", inplace = True)
        elif(j[0:2] == 'SC'):
            test1['Ticket'].replace(to_replace = j, value ="A", inplace = True)
        elif(j[0:2] == 'PP'):
            test1['Ticket'].replace(to_replace = j, value ="P", inplace = True)
        else:
            test1['Ticket'].replace(to_replace = j, value ="S", inplace = True)
    elif(len(j) == 4):
        if(j[0] == '9' or j[0] == '3' or j[0] == '4'):
            test1['Ticket'].replace(to_replace = j, value ="P", inplace = True)
        else:
            test1['Ticket'].replace(to_replace = j, value ="CA", inplace = True)

g = list(range(11600,12000,1))
p = list(range(12000,20000,1))
c = list(range(20000,30000,1))
a = list(range(30000,70000,1))             
for i in test1['Ticket']:
    j = str(i)
    if(len(j) == 5):
        if int(j) in g:
            test1['Ticket'].replace(to_replace = j, value ="G", inplace = True)
        elif(int(j) in p):
            test1['Ticket'].replace(to_replace = j, value ="PC", inplace = True)
        elif(int(j) in c):
            test1['Ticket'].replace(to_replace = j, value ="CA", inplace = True)
        elif(int(j) in a):
            test1['Ticket'].replace(to_replace = j, value ="A", inplace = True)

for i in test1['Ticket']:
    j = str(i)
    if(len(j) > 2):
        if(j[0]=='C' or j[0]=='W'):
            test1['Ticket'].replace(to_replace = j, value ="CA", inplace = True)
        elif(j[0]== 'A'):
            test1['Ticket'].replace(to_replace = j, value ="A", inplace = True)
        elif(j[0:2] == 'SC'):
            test1['Ticket'].replace(to_replace = j, value ="A", inplace = True)
        
          
for i in test1['Ticket']:
    j = str(i)
    if(len(j) > 3):
        if(j[0:3] == 'S.C'):
            test1['Ticket'].replace(to_replace = j, value ="A", inplace = True)
        if(j[0:2] == 'PC'):
            test1['Ticket'].replace(to_replace = j, value ="PC", inplace = True)
        if(j[0:4] == 'STON' or j[0:5] == 'SOTON'):
            test1['Ticket'].replace(to_replace = j, value ="S", inplace = True)
        if(j[0] == 'P' or j[0:2] == 'SW' or j[0:3] == 'S.W'):
            test1['Ticket'].replace(to_replace = j, value ="P", inplace = True)
        if(j[0] == 'F'):
            test1['Ticket'].replace(to_replace = j, value ="F", inplace = True)
        else:
            test1['Ticket'].replace(to_replace = j, value ="SOC", inplace = True)
           
for i in test1['Ticket']:
    j = str(i)
    if j.isnumeric():
        test1['Ticket'].replace(to_replace = j, value ="A", inplace = True)      

test1.columns
test2 = pd.get_dummies(test1, drop_first = True)   
sns.heatmap(test2.isnull(), cbar=False)
m = test2['Fare'].mean()
n = test2['Fare'].median()
test2['Fare'].fillna(n , inplace = True)
g = pd.read_csv('C:\\Users\\Shriyash Shende\\Desktop\\titanic\\gender_submission.csv')
pred1 = clf.predict(test2)                 
 
cn1 = confusion_matrix(g['Survived'], pred1)        

per1 = cn1[0,0] + cn1[1,1]
p1 = per1 + cn1[0,1] + cn1[1,0]
(per1 / p1) * 100


import pickle
from sklearn.ensemble import RandomForestClassifier         
clf1 = RandomForestClassifier()        
clf1.fit(X_train, Y_train)            
pred_f = clf1.predict(X_test)            
cn2 = confusion_matrix(Y_test, pred_f)         
per2 = cn2[0,0] + cn2[1,1]
p2 = per2 + cn2[0,1] + cn2[1,0]
(per2 / p2) * 100 

pred_f1 = clf1.predict(test2)  
cn3 = confusion_matrix(g['Survived'], pred_f1)        

per3 = cn3[0,0] + cn3[1,1]
p3 = per3 + cn3[0,1] + cn3[1,0]
(per3 / p3) * 100  

# Saving model to disk
test2.columns
pickle.dump(clf1, open('C:\\Users\\Shriyash Shende\\Desktop\\titanic\\deployment\\model.pkl','wb'))

pred1
pred_f1

pL = pd.DataFrame(pred1)
pL.rename(columns={0:'Survived'}, inplace = True)
result = pd.concat([g['PassengerId'], pL], axis=1)
result.to_csv('C:\\Users\\Shriyash Shende\\Desktop\\titanic\\Submission.csv')

pL1 = pd.DataFrame(pred_f1)
pL1.rename(columns={0:'Survived'}, inplace = True)
result1 = pd.concat([g['PassengerId'], pL], axis=1)
result1.to_csv('C:\\Users\\Shriyash Shende\\Desktop\\titanic\\Submission1.csv')
