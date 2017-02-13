#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 12:43:04 2017

@author: marleneguraieb
"""

#%reset
import glob
import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
#import types
#import warnings
#import itertools

pd.set_option('display.max_colwidth', -1)
pd.options.display.max_columns = None

#First I import all their data as a dictionary of datasets from which I will 
#extract the features for different models. The files are all in the dicrectory
# /raw_data.

path =r'raw_data' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
databases = {}
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0,low_memory=False)
    databases[re.sub(r'\W.*','',re.sub(r'.*/', '', file_))] = df
              
#Every item has two identifiers, first (unique) item_id, and then which layer
#of the triangle it belongs to.

item_id = databases['triangleLayerItems']._id
layer_id = databases['triangleLayerItems'].triangleLayerId
data = pd.concat([item_id,layer_id],axis=1)

#An instance of this data (an item) is a business objective, and the model 
#will try to predict whether the objective will be completed or not by 
#extracting various types of features from the natural language description of
# the objective and the non-text data associated with it in the CrossLead 
#database. The dataset starts with a unique identifyier for the 
#objective, called _id

data['fin_status'] = databases['triangleLayerItems'].statuses.str.replace(r'.*"percent":', '').str.replace(r',.*','')
data['fin_status'] = pd.to_numeric(data['fin_status'],errors='coerce')

#Visualization of target variable:
plt.hist(data['fin_status'].dropna(),color='blue',range=(0, 100))
plt.title("Completed status")
plt.xlabel("Percent complete")
plt.ylabel("Frequency")
plt.show()

#Plus a large number of missing values:
data['fin_status'].isnull().sum()

data.loc[data.loc[:,'fin_status'].isnull()==True,'completed'] = int(1)
data.loc[(data.loc[:,'fin_status']==0),'completed'] = 2
data.loc[(data.loc[:,'fin_status']>0) & (data.loc[:,'fin_status']<=50),'completed'] = int(3)
data.loc[(data.loc[:,'fin_status']>50),'completed'] = int(4)

data.completed = data.completed.astype(int)

print(data.completed.value_counts())

#This is our text variable, all text features will be extracted from the 
#objective name

data['item_name'] = databases['triangleLayerItems'].name
    
# Now I extract structured features from dataset
#Watchers
data['watchers'] = [str(element).count('oid') for element in databases['triangleLayerItems'].watchers]
print(data.watchers.value_counts())

data['dependenciesN'] = [str(element).count('created') for element in databases['triangleLayerItems'].dependencies]
#print(data['dependenciesN'].value_counts())
    
#Labels:

TLIlabels = sorted([item.lower() for item in databases['triangleLayers'].label.unique()])

TLImap = ['objective','other','other','strategy','principle','principle','initiative','initiative',
          'initiative','metric','metric','metric','metric','metric','mission','mission','objective',
          'objective','mission','value','vision','metric','value','other','other','other','objective',
          'principle','strategy','strategy','strategy','strategy','metric','metric','metric','other','other',
          'mission','value','vision','vision','vision','vision']

labelDict = {}
for label,value in zip(TLIlabels,TLImap):
    labelDict[label] = value
             
gen = data.iterrows()
labels = []
order = []
for index,row in gen:
    try:
        labels.append(databases['triangleLayers'].loc[(databases['triangleLayers']._id == row['triangleLayerId']),
                                                      'label'].item())
    except:
        labels.append(np.nan)
    try:
        order.append(databases['triangleLayers'].loc[(databases['triangleLayers']._id == row['triangleLayerId']),
                                                      'order'].item())
    except:
        order.append(np.nan)

data['labels'] = labels
data['labels'] = data['labels'].str.lower().map(labelDict)

print(data['labels'].value_counts())

#make new dictionary:
labelDict_comp = {'initiative':'initiative', 'strategy':'strategy', 'objective':'objective', 
                  'vision':'pvvm', 'principle':'pvvm',
       'metric':'metric', 'other':'other', 'mission':'pvvm', 'value':'pvvm'}

data['labels'] = data['labels'].map(labelDict_comp)
data['labels'] = data['labels'].fillna('')
print(data['labels'].value_counts())

data['order'] = order
data['userId'] = databases['triangleLayerItems'].userId
    
label_ranks = pd.get_dummies(data['labels'].astype('category').values, prefix='label')
label_ranks.columns

data = data.drop('labels',axis=1).join(label_ranks.ix[:, 'label_initiative':])

# Get scores for owner of the objective
users = databases['users']
users = users[['_id','emailLower']]
scores = pd.read_csv('user_scores.csv')

# append scores to users:
gen = users.iterrows()
Info = []
Energy = []
Access = []
for index,row in gen:
    try:
        Info.append(scores.loc[(scores.Email == row['emailLower']),'Info'].item())
    except:
        Info.append(np.nan)
    try:
        Energy.append(scores.loc[(scores.Email == row['emailLower']),'Energy'].item())
    except:
        Energy.append(np.nan)
    try:
        Access.append(scores.loc[(scores.Email == row['emailLower']),'Access'].item())
    except:
        Access.append(np.nan)
        
users.loc[:,'Info'] = Info
users.loc[:,'Energy'] = Energy
users.loc[:,'Access'] = Access
         
# Now append user data to my data:
gen = data.iterrows()
Info = []
Energy = []
Access = []
for index,row in gen:
    try:
        Info.append(users.loc[(users._id == row['userId']),'Info'].item())
    except:
        Info.append(np.nan)
    try:
        Energy.append(users.loc[(users._id == row['userId']),'Energy'].item())
    except:
        Energy.append(np.nan)
    try:
        Access.append(users.loc[(users._id == row['userId']),'Access'].item())
    except:
        Access.append(np.nan)

data['Info'] = Info
data['Energy'] = Energy
data['Access'] = Access
    
data = data[['watchers','dependenciesN'] + 
            list(label_ranks.columns[1:].values) + 
            ['order','Info','Energy','Access','item_name','completed']]

data = data.dropna(subset=['item_name'], how='all')

data = data.fillna(data.mean())

print('Description of the dataset')
print('_____________________________________________________________________')
print(data.describe())

data_text = data[['item_name','completed']]
data_text.columns = ['X','Y']

data_feat = data.drop('item_name',axis=1)
data_feat.rename(columns={'completed':'Y'}, inplace=True)
    