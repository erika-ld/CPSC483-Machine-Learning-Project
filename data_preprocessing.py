#!/usr/bin/env python
# coding: utf-8

# # CPSC483-06 #
# # Jupyter Notebook for dataset preprocessing of 'dreams' dataset #
# Dulce Funez Chinchilla, Drashti Mehta, Erika Dickson
# 

# In[128]:


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
from sklearn import preprocessing


# In[129]:


#Read the csv into a pandas dataframe and examine the raw dataset
dreams = pd.read_csv('dreams_dataset.csv')
print(dreams.shape)
dreams.head(10)


# In[130]:


#Add an additional column 'Gender' for the Gender of the dreamer, place it in index 2, intialize with 'n/a'
#Column 'Gender' will be used for data analysis and later conclusion drawing
dreams.insert(2, 'Gender', 'n/a')


# In[131]:


#Update the Gender column to have dreamer Gender as numeric binary
#Female: 1
#Male: 0
dreamer_gender = {
    'alta' : '1',
    'angie' : '1',
    'arlie' : '1',
    'b' : '1',
    'b2' : '1',
    'bay_area_girls_456' : '1',
    'bay_area_girls_789' : '1',
    'bea1' : '1',
    'bea2' : '1',
    'blind-f' : '1',
    'blind-m' : '0',
    'bosnak' : '0',
    'chris' : '0',
    'chuck' : '0',
    'dahlia' : '1',
    'david' : '0',
    'dorothea' : '1',
    'ed' : '0',
    'edna' : '1',
    'elizabeth' : '1',
    'emma' : '1',
    'emmas_husband' : '0',
    'esther' : '1',
    'hall_female' : '1',
    'norms-f' : '1',
    'izzy' : '1',
    'jasmine1' : '1',
    'jasmine2' : '1',
    'jasmine3' : '1',
    'jasmine4' : '1',
    'jeff' : '0',
    'joan' : '1',
    'kenneth' : '0',
    'lawrence' : '0',
    'mack' : '0',
    'madeline1-hs' : '1',
    'madeline2-dorms' : '1',
    'madeline3-offcampus' : '1',
    'madeline4-postgrad' : '1',
    'mark' : '0',
    'melissa' : '1',
    'melora' : '1',
    'melvin' : '0',
    'merri' : '1',
    'miami-home' : '0',
    'miami-lab' : '0',
    'midwest_teens-f' : '1',
    'midwest_teens-m' : '0',
    'nancy' : '1',
    'natural_scientist' : '0',
    'norman' : '0',
    'wedding' : '1',
    'norms-m' : '0',
    'pegasus' : '0',
    'peru-f' : '1',
    'peru-m' : '0',
    'phil1' : '0',
    'phil2' : '0',
    'phil3' : '0',
    'physiologist' : '0',
    'ringo' : '0',
    'samantha' : '1',
    'seventh_graders' : '1',
    'toby' : '0',
    'tom' : '0',
    'ucsc_women' : '1',
    'vickie' : '1',
    'vietnam_vet' : '0',
    'vietnam_vet2' : '0',
    'west_coast_teens' : '1',
}

#Populate the df 'Gender'column with the transformed values
for key, val in dreamer_gender.items():
    dreams.loc[dreams['dreamer'] == key, 'Gender'] = val

dreams.head(5)

dreams['Gender'] = dreams['Gender'].astype(float)
dreams['Gender'].dtype



# In[132]:


#Split train & test set with 80:20 ratio 
X = dreams.iloc[:, :-1]
y = dreams.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Make a copy of the raw train and test sets for later use
X_train_raw = X_train.copy()
y_train_raw = y_train.copy()
X_test_raw = X_test.copy()
y_test_raw = y_test.copy()


# Proceed to data preprocessing on train set 

# In[133]:


#Handle missing values: find how many columns have missing values, and how many many missing values
X_train.isna().sum()


# In[134]:


#Dimensionality reduction
#Remove unnecessary columns from the train set that are unusable or irrelevant
X_train.drop(' dream_id', axis=1, inplace=True)
X_train.drop('dream_language', axis=1, inplace=True)
X_train.drop('dream_date', axis=1, inplace=True)
X_train.drop('dreamer', axis=1, inplace=True)
X_train.drop('description', axis=1, inplace=True)
X_train.drop('text_dream', axis=1, inplace=True)

#Remove unnecessary/unusable columns with missing values 
X_train.drop('characters_code', axis=1, inplace=True)
X_train.drop('emotions_code', axis=1, inplace=True)
X_train.drop('aggression_code', axis=1, inplace=True)
X_train.drop('friendliness_code', axis=1, inplace=True)
X_train.drop('sexuality_code', axis=1, inplace=True)


# In[135]:


#Checking none of the remaining columns have missing values
X_train.isna().sum()


# Discover & visualize the cleaned data to gain insights

# In[136]:


#Discovering & ensuring that all the key feature columns are in the same range of values
print('Column "Aggression/Friendliness" minimum value:', X_train['Aggression/Friendliness'].min())
print('Column "Aggression/Friendliness" maximum value:', X_train['Aggression/Friendliness'].max())
print('Column "A/CIndex" minimum value:', X_train['A/CIndex'].min())
print('Column "A/CIndex" maximum value:', X_train['A/CIndex'].max())
print('Column "F/CIndex" minimum value:', X_train['F/CIndex'].min())
print('Column "F/CIndex" maximum value:', X_train['F/CIndex'].max())
print('Column "S/CIndex" minimum value:', X_train['S/CIndex'].min())
print('Column "S/CIndex" maximum value:', X_train['S/CIndex'].max())
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


# In[137]:


#Need to scale within a range of 0-1: A/CIndex, F/CIndex, S/CIndex
scaler = preprocessing.MinMaxScaler()
X_train['A/CIndex'] = scaler.fit_transform(X_train[['A/CIndex']])
X_train['F/CIndex'] = scaler.fit_transform(X_train[['F/CIndex']])
X_train['S/CIndex'] = scaler.fit_transform(X_train[['S/CIndex']])

#Check that the data has been scaled between 0-1
print('Column "A/CIndex" minimum value:', X_train['A/CIndex'].min())
print('Column "A/CIndex" maximum value:', X_train['A/CIndex'].max())
print('Column "F/CIndex" minimum value:', X_train['F/CIndex'].min())
print('Column "F/CIndex" maximum value:', X_train['F/CIndex'].max())
print('Column "S/CIndex" minimum value:', X_train['S/CIndex'].min())
print('Column "S/CIndex" maximum value:', X_train['S/CIndex'].max())

X_train.head(10)


# Visualizing the data with scatterplots and histograms to determine relationships between the variables & target feature, and frequencies of the attributes

# In[138]:


plt.xlabel("Gender")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Gender'], y_train, color = 'orange')
plt.show()


# In[139]:


plt.xlabel("Aggression/Friendliness")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Aggression/Friendliness'], y_train, color = 'gray')
plt.show()


# In[140]:


plt.xlabel("A/CIndex")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['A/CIndex'], y_train, color = 'indigo')
plt.show()


# In[141]:


plt.xlabel("F/CIndex")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['F/CIndex'], y_train, color = 'seagreen')
plt.show()


# In[142]:


plt.xlabel("S/CIndex")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['S/CIndex'], y_train, color = 'cornflowerblue')
plt.show()


# In[143]:


plt.xlabel("Male")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Male'], y_train, color = 'red')
plt.show()


# In[144]:


plt.xlabel("Animal")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Animal'], y_train, color = 'blue')
plt.show()


# In[145]:


plt.xlabel("Friends")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Friends'], y_train, color = 'green')
plt.show()


# In[146]:


plt.xlabel("Family")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Family'], y_train, color = 'yellow')
plt.show()


# In[147]:


plt.xlabel("Dead&Imaginary")
plt.ylabel("Negative Emotions")
plt.scatter(X_train['Dead&Imaginary'], y_train, color = 'purple')
plt.show()


# In[148]:


plt.hist(X_train['Gender'], bins=10, color='orange', edgecolor='black')  
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()


# In[149]:


plt.hist(X_train['Aggression/Friendliness'], bins=10, color='gray', edgecolor='black')  
plt.xlabel('Aggression/Friendliness')
plt.ylabel('Frequency')
plt.show()


# In[150]:


plt.hist(X_train['A/CIndex'], bins=10, color='indigo', edgecolor='black')  
plt.xlabel('A/CIndex')
plt.ylabel('Frequency')
plt.show()


# In[151]:


plt.hist(X_train['F/CIndex'], bins=10, color='seagreen', edgecolor='black')  
plt.xlabel('F/CIndex')
plt.ylabel('Frequency')
plt.show()


# In[152]:


plt.hist(X_train['S/CIndex'], bins=10, color='cornflowerblue', edgecolor='black')  
plt.xlabel('S/CIndex')
plt.ylabel('Frequency')
plt.show()


# In[153]:


plt.hist(X_train['Male'], bins=10, color='red', edgecolor='black')  
plt.xlabel('Male')
plt.ylabel('Frequency')
plt.show()


# In[154]:


plt.hist(X_train['Animal'], bins=10, color='blue', edgecolor='black')  
plt.xlabel('Animal')
plt.ylabel('Frequency')
plt.show()


# In[155]:


plt.hist(X_train['Friends'], bins=10, color='green', edgecolor='black') 
plt.xlabel('Friends')
plt.ylabel('Frequency')
plt.show()


# In[156]:


plt.hist(X_train['Family'], bins=10, color='yellow', edgecolor='black') 
plt.xlabel('Family')
plt.ylabel('Frequency')
plt.show()


# In[157]:


plt.hist(X_train['Dead&Imaginary'], bins=10, color='purple', edgecolor='black')  
plt.xlabel('Dead&Imaginary')
plt.ylabel('Frequency')
plt.show()


# Checking for existing correlations between the attributes and target feature
# Determining the degree to which they are correlated and which features have the highest correlation to each other & target feature

# In[158]:


# Correlation matrix
selected_columns = X_train[['Gender', 'Male', 'Animal', 'Friends', 'Family', 'Dead&Imaginary', 
                            'Aggression/Friendliness', 'A/CIndex', 'F/CIndex', 'S/CIndex']]

selected_columns['NegativeEmotions'] = y_train
selected_columns.head(10)


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

# In[159]:


training_df = X_train.copy()
training_df['NegativeEmotions'] = y_train
print(training_df.columns)
display(training_df.head())
training_df.to_csv("training_data.csv")


# Reflect changes on training set to test set

# In[160]:


#Drop columns from test set to match the training set
X_test.drop(' dream_id', axis=1, inplace=True)
X_test.drop('dreamer', axis=1, inplace=True)
X_test.drop('description', axis=1, inplace=True)
X_test.drop('text_dream', axis=1, inplace=True)
X_test.drop('dream_language', axis=1, inplace=True)
X_test.drop('dream_date', axis=1, inplace=True)
X_test.drop('characters_code', axis=1, inplace=True)
X_test.drop('emotions_code', axis=1, inplace=True)
X_test.drop('aggression_code', axis=1, inplace=True)
X_test.drop('friendliness_code', axis=1, inplace=True)
X_test.drop('sexuality_code', axis=1, inplace=True)

#Scale columns
scaler = preprocessing.MinMaxScaler()
X_test['A/CIndex'] = scaler.fit_transform(X_test[['A/CIndex']])
X_test['F/CIndex'] = scaler.fit_transform(X_test[['F/CIndex']])
X_test['S/CIndex'] = scaler.fit_transform(X_test[['S/CIndex']])


# In[161]:


#Make a copy of the files for later processing in subsequent files
X_train_copy = X_train.copy()
y_train_copy = y_train.copy()
X_test_copy = X_test.copy()
y_test_copy = y_test.copy()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script data_preprocessing.ipynb')

