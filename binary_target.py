#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# read the csv
dreamset = pd.read_csv('dreams_dataset.csv')

dreamset.head()


# In[ ]:


#Removes unnecessary columns or features
columns_to_remove = [' dream_id', 'dreamer', 'description', 'dream_date', 'dream_language', 'text_dream', 'characters_code', 'emotions_code', 'aggression_code', 'friendliness_code', 'sexuality_code','Male', 'Animal', 'Friends', 'Family'
             , 'Dead&Imaginary', 'S/CIndex'] 
dreamset = dreamset.drop(columns=columns_to_remove)

dreamset.head()


# In[ ]:


# Check the correlation of the features and target label
correlation_matrix = dreamset.corr()

print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


# Prints the type of the variables
print(dreamset.dtypes)


# In[ ]:


# This cell checks the min, max, and range of the features and target variable
variables = [' dream_id', 'dreamer', 'description', 'dream_date', 'dream_language', 'text_dream', 'characters_code'
             , 'emotions_code', 'aggression_code', 'friendliness_code', 'sexuality_code', 'Male', 'Animal', 'Friends', 'Family'
             , 'Dead&Imaginary', 'Aggression/Friendliness', 'A/CImndex', 'F/CIndex/', 'S/CIndex', 'NegativeEmotions']
variable = ['Aggression/Friendliness', 'A/CIndex', 'F/CIndex', 'NegativeEmotions']
for column_name in variable:
    min_value = dreamset[column_name].min()
    max_value = dreamset[column_name].max()
    range_value = max_value - min_value
    print(f"Min {min_value}")
    print(f"Max: {max_value}")
    print(f"Range of {column_name}: {range_value}")


# In[ ]:


# Change the target label from float to binary
dreamset['NegativeEmotions'] = dreamset['NegativeEmotions'].apply(lambda x: 1 if x > 0.5 else 0)

print(dreamset.head())


# In[ ]:


# Recheck correlation now that the target is binary
correlation_matrix = dreamset.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


#Uses KNN to predict the target label and checks accuracy of the model
X = dreamset.drop('NegativeEmotions', axis=1)
y = dreamset['NegativeEmotions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy of the KNN model: {accuracy * 100:.2f}%')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script binary_target.ipynb')

