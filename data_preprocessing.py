#!/usr/bin/env python
# coding: utf-8

# # CPSC483-06 #
# # Jupyter Notebook for dataset preprocessing of 'dreams' dataset #
# Dulce Funez Chinchilla, Drashti Mehta, Erika Dickson
# 

# In[99]:


#Import statements
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from scipy import stats


# In[100]:


dreams = pd.read_csv('dreams_dataset.csv')
print(dreams.shape)
dreams.head(10)


# In[101]:


#Add an additional column 'gender' for the gender of the dreamer, place it in index 2, intialize with 'n/a'
#Column 'gender' will be used for data analysis and later conclusion drawing
dreams.insert(2, 'gender', 'n/a')


# In[102]:


#Update the gender column to have dreamer gender
dreamer_gender = {
    'alta' : 'F',
    'angie' : 'F',
    'arlie' : 'F',
    'b' : 'F',
    'b2' : 'F',
    'bay_area_girls_456' : 'F',
    'bay_area_girls_789' : 'F',
    'bea1' : 'F',
    'bea2' : 'F',
    'blind-f' : 'F',
    'blind-m' : 'M',
    'bosnak' : 'M',
    'chris' : 'M',
    'chuck' : 'M',
    'dahlia' : 'F',
    'david' : 'M',
    'dorothea' : 'F',
    'ed' : 'M',
    'edna' : 'F',
    'elizabeth' : 'F',
    'emma' : 'F',
    'emmas_husband' : 'M',
    'esther' : 'F',
    'hall_female' : 'F',
    'norms-f' : 'F',
    'izzy' : 'F',
    'jasmine1' : 'F',
    'jasmine2' : 'F',
    'jasmine3' : 'F',
    'jasmine4' : 'F',
    'jeff' : 'M',
    'joan' : 'F',
    'kenneth' : 'M',
    'lawrence' : 'M',
    'mack' : 'M',
    'madeline1-hs' : 'F',
    'madeline2-dorms' : 'F',
    'madeline3-offcampus' : 'F',
    'madeline4-postgrad' : 'F',
    'mark' : 'M',
    'melissa' : 'F',
    'melora' : 'F',
    'melvin' : 'M',
    'merri' : 'F',
    'miami-home' : 'M',
    'miami-lab' : 'M',
    'midwest_teens-f' : 'F',
    'midwest_teens-m' : 'M',
    'nancy' : 'F',
    'natural_scientist' : 'M',
    'norman' : 'M',
    'wedding' : 'F',
    'norms-m' : 'M',
    'pegasus' : 'M',
    'peru-f' : 'f',
    'peru-m' : 'm',
    'phil1' : 'm',
    'phil2' : 'm',
    'phil3' : 'm',
    'physiologist' : 'M',
    'ringo' : 'M',
    'samantha' : 'F',
    'seventh_graders' : 'F',
    'toby' : 'M',
    'tom' : 'M',
    'ucsc_women' : 'F',
    'vickie' : 'F',
    'vietnam_vet' : 'M',
    'vietnam_vet2' : 'M',
    'west_coast_teens' : 'F',
}

for key, val in dreamer_gender.items():
    dreams.loc[dreams['dreamer'] == key, 'gender'] = val

dreams.head(5)


# In[103]:


#Split train & test set with 80:20 ratio 

X = dreams.iloc[:, :-1]
y = dreams.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Proceed to data preprocessing on train set 

# In[104]:


#Handle missing values: find how many columns have missing values, and how many many missing values
X_train.isna().sum()


# In[105]:


#Dimensionality reduction
#Remove unnecessary columns from the train set
X_train.drop('dream_language', axis=1, inplace=True)
X_train.drop('dream_date', axis=1, inplace=True)

#Remove unnecessary columns with missing values 
X_train.drop('characters_code', axis=1, inplace=True)
X_train.drop('emotions_code', axis=1, inplace=True)
X_train.drop('aggression_code', axis=1, inplace=True)
X_train.drop('friendliness_code', axis=1, inplace=True)
X_train.drop('sexuality_code', axis=1, inplace=True)
X_train.head(10)


# In[106]:


#Checking none of the remaining columns have missing values
X_train.isna().sum()


# In[107]:


#Checking for any duplicate rows
len(X_train) - len(X_train.drop_duplicates())


# Discover & visualize the cleaned data to gain insights

# In[108]:


#Discovering & ensuring that all the key feature columns are in the same range of values
print('Column "Male" minimum value:', X_train['Male'].min())
print('Column "Male" maximum value:', X_train['Male'].max())
print('Column "Animal" minimum value:', X_train['Animal'].min())
print('Column "Animal" maximum value:', X_train['Animal'].max())
print('Column "Friends" minimum value:', X_train['Friends'].min())
print('Column "Friends" maximum value:', X_train['Friends'].max())
print('Column "Family" minimum value:', X_train['Family'].min())
print('Column "Family" maximum value:', X_train['Family'].max())
print('Column "Dead&Imaginary" minimum value:', X_train['Dead&Imaginary'].min())
print('Column "Dead&Imaginary" maximum value:', X_train['Dead&Imaginary'].max())
print('Column "NegativeEmotions" minimum value:', y_train.min())
print('Column "NegativeEmotions" maximum value:', y_train.max())


# In[109]:


plt.xlabel("Male")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Male'], y_train, color = 'red')
plt.show()


# In[110]:


plt.xlabel("Animal")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Animal'], y_train, color = 'blue')
plt.show()


# In[111]:


plt.xlabel("Friends")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Friends'], y_train, color = 'green')
plt.show()


# In[112]:


plt.xlabel("Family")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Family'], y_train, color = 'yellow')
plt.show()


# In[113]:


plt.xlabel("Dead&Imaginary")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Dead&Imaginary'], y_train, color = 'purple')
plt.show()


# In[116]:


plt.hist(X_train['Male'], bins=10, color='red', edgecolor='black')  
plt.xlabel('Male')
plt.ylabel('Frequency')
plt.show()


# In[117]:


plt.hist(X_train['Animal'], bins=10, color='blue', edgecolor='black')  
plt.xlabel('Animal')
plt.ylabel('Frequency')
plt.show()


# In[119]:


plt.hist(X_train['Friends'], bins=10, color='green', edgecolor='black') 
plt.xlabel('Friends')
plt.ylabel('Frequency')
plt.show()


# In[120]:


plt.hist(X_train['Family'], bins=10, color='yellow', edgecolor='black') 
plt.xlabel('Family')
plt.ylabel('Frequency')
plt.show()


# In[121]:


plt.hist(X_train['Dead&Imaginary'], bins=10, color='purple', edgecolor='black')  
plt.xlabel('Dead&Imaginary')
plt.ylabel('Frequency')
plt.show()


# Checking for correlations

# In[115]:


#correlation matrix
selected_columns = dreams[['Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary', 'NegativeEmotions']]

# Calculate the correlation matrix
correlation_matrix = selected_columns.corr()

# Display the correlation matrix
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# Exporting the processed training data frame to a csv file

# In[152]:


training_df = X_train.copy()
training_df['NegativeEmotions'] = y_train
print(training_df.columns)
display(training_df.head())
training_df.to_csv("training_data.csv")


# Reflect changes on training set to test set

# In[153]:


#Drop columns from test set to match the training set
X_test.drop('dream_language', axis=1, inplace=True)
X_test.drop('dream_date', axis=1, inplace=True)
X_test.drop('characters_code', axis=1, inplace=True)
X_test.drop('emotions_code', axis=1, inplace=True)
X_test.drop('aggression_code', axis=1, inplace=True)
X_test.drop('friendliness_code', axis=1, inplace=True)
X_test.drop('sexuality_code', axis=1, inplace=True)

