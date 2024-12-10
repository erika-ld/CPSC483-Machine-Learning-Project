#!/usr/bin/env python
# coding: utf-8

# Team Members: Erika Dickson, Drashti Mehta, Dulce Funez Chinchilla

# In[62]:


import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from data_preprocessing import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import xgboost as xg 
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.ensemble import HistGradientBoostingRegressor


# In[58]:


#Read into a pandas dataframe the training_data.csv made from the data_preprocessing notebook
dreams_proc = pd.read_csv('training_data.csv')
print(dreams_proc.shape)
dreams_proc.head(10)


# In[59]:


#Use scikit learn's KNN regression function
#KNN Regression on the processed dataset w/ k = 10
knn_regressor = KNeighborsRegressor(n_neighbors=10)
knn_regressor.fit(X_train_copy, y_train_copy)
y_predict = knn_regressor.predict(X_test_copy)

mse = mean_squared_error(y_test_copy, y_predict)
print("KNN MSE:", mse)

filename = 'finalized_model_M1.model'
pickle.dump(knn_regressor, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result_knn = loaded_model.score(X_test_copy, y_test_copy)
print("KNN Regressor R^2 Score:", result_knn)


# In[60]:


#Use scikit learn's linear regression function
#Linear regression on the processed dataset
linear_regression = LinearRegression() 
linear_regression.fit(X_train_copy, y_train_copy) 
y_predict = linear_regression.predict(X_test_copy)

mse_lr = mean_squared_error(y_test_copy, y_predict)
print("Linear Regression MSE:", mse_lr)

filename_lr = 'finalized_model_LR1.model'
pickle.dump(linear_regression, open(filename_lr, 'wb'))

loaded_lr_model = pickle.load(open(filename_lr, 'rb'))
result_lr = loaded_lr_model.score(X_test_copy, y_test_copy)
print("Linear Regression R^2 Score:", result_lr)


# In[63]:


#Use scikit learn's SVM Regressor function
#SVM regression on the processed dataset
svm_regressor = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_regressor.fit(X_train_copy, y_train_copy)
y_predict_svm = svm_regressor.predict(X_test_copy)

mse_svm = mean_squared_error(y_test_copy, y_predict_svm)
print("SVM MSE:", mse_svm)

filename_svm = 'finalized_model_SVM1.model'
pickle.dump(svm_regressor, open(filename_svm, 'wb'))

loaded_svm_model = pickle.load(open(filename_svm, 'rb'))
result_svmr = loaded_svm_model.score(X_test_copy, y_test_copy)
print("SVM R^2 Score:", result_svmr)


# In[64]:


#Use scikit learn's random forest function
#Random forest on the processed dataset
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_copy, y_train_copy)
y_predict_rf = rf_regressor.predict(X_test_copy)

mse_rf = mean_squared_error(y_test_copy, y_predict_rf)
print("Random Forest MSE :", mse_rf)

filename_rf = 'finalized_model_RF1.model'
pickle.dump(rf_regressor, open(filename_rf, 'wb'))

loaded_rf_model = pickle.load(open(filename_rf, 'rb'))

result_rfr = loaded_rf_model.score(X_test_copy, y_test_copy)
print("RF regressor Score:", result_rfr)


# In[65]:


#Use scikit learn's Histogram-based Gradient Boosting Regression Tree
#HistGradientBoostingRegressor on the processed dataset
hgbr = HistGradientBoostingRegressor(loss = 'squared_error', learning_rate=0.1, max_iter=100,
                                    max_bins=255, early_stopping='auto', n_iter_no_change=100).fit(X_train_copy, y_train_copy)
y_predict_hgbr = hgbr.predict(X_test_copy)

mse_hgbr = mean_squared_error(y_test_copy, y_predict_hgbr)
print("HistGradientBoostingRegression MSE:", mse_hgbr)

filename_hgbr = 'finalized_model_HGBR1.model'
pickle.dump(hgbr, open(filename_hgbr, 'wb'))

loaded_hgbr_model = pickle.load(open(filename_hgbr, 'rb'))
result_hgbr = loaded_hgbr_model.score(X_test_copy, y_test_copy)
print("Histogram-based Gradient Boosting Regression Tree R^2 Score:", result_hgbr)


# In[66]:


#Extreme Gradient Boosting
#XGB on the processed dataset
xgb_r = xg.XGBRegressor(objective ='reg:squarederror', n_estimators = 10, seed = 123) 
xgb_r.fit(X_train_copy, y_train_copy) 
y_predict_xgb = xgb_r.predict(X_test_copy) 

mse_xgb = mean_squared_error(y_test_copy, y_predict_xgb)
print("XGBoost MSE:", mse_xgb)

filename_xgb = 'finalized_model_XGB1.model'
pickle.dump(xgb_r, open(filename_xgb, 'wb'))

loaded_xgb_model = pickle.load(open(filename_xgb, 'rb'))
result_xgbr = xgb_r.score(X_test_copy, y_test_copy)
print("XGBoost Score:", result_xgbr)


# In[67]:


#Use scikit learn's Elastic Net Regression function
#Elastic net regressor on the processed dataset
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train_copy, y_train_copy)
y_pred_elasnet = elastic_net.predict(X_test_copy)

filename_elasnet = 'finalized_model_EN1.model'
pickle.dump(elastic_net, open(filename_elasnet, 'wb'))

mse_en = mean_squared_error(y_test_copy, y_pred_elasnet)
print("Elastic Net Regression MSE:", mse_en)

loaded_elasnet_model = pickle.load(open(filename_elasnet, 'rb'))
result_en = loaded_elasnet_model.score(X_test_copy, y_test_copy)
print("Elastic Net Regression R^2 Score:", result_en)


# In[68]:


#Use scikit learn's Lasso Regression function
#Lasso regressor on the processed dataset
lasso = Lasso(alpha=0.1)  # Regularization strength (higher alpha means more regularization)
lasso.fit(X_train_copy, y_train_copy)
y_pred_lasso = lasso.predict(X_test_copy)

mse_l = mean_squared_error(y_test_copy, y_pred_lasso)
print(f"Mean Squared Error: {mse_l:.4f}")

filename_lasso = 'finalized_model_Lasso1.model'
pickle.dump(lasso, open(filename_lasso, 'wb'))

loaded_lasso_model = pickle.load(open(filename_lasso, 'rb'))
result_lasso = loaded_lasso_model.score(X_test_copy, y_test_copy)
print("Lasso Regression R^2 Score:", result_lasso)

print("\nLasso Coefficients:")
print(lasso.coef_)

non_zero_coefficients = np.where(lasso.coef_ != 0)[0]
print(f"\nSelected features (non-zero coefficients): {non_zero_coefficients}") 


# In[69]:


#Use scikit learn's Huber Regression function
#Huber regressor on the processed dataset
huber = HuberRegressor().fit(X_train_copy, y_train_copy)
y_predict_huber = huber.predict(X_test_copy)

mse_huber = mean_squared_error(y_test_copy, y_predict_huber)
print("Huber Regressor MSE:", mse_huber)

filename_huber = 'finalized_model_H1.model'
pickle.dump(huber, open(filename_huber, 'wb'))

loaded_huber_model = pickle.load(open(filename_huber, 'rb'))
result_huber = loaded_huber_model.score(X_test_copy, y_test_copy)
print("Huber Regressor R^2 Score:", result_huber)


# In[70]:


#Use scikit learn's Ridge Regression function
#Ridge regression on the processed dataset
rdg = Ridge(alpha = 1.0)
rdg.fit(X_train_copy, y_train_copy)
y_predict_rdg = rdg.predict(X_test_copy)

mse_rdg = mean_squared_error(y_test_copy, y_predict_rdg)
print("Ridge Regressor MSE:", mse_rdg)

filename_rdg = 'finalized_model_R1.model'
pickle.dump(rdg, open(filename_rdg, 'wb'))

loaded_rdg_model = pickle.load(open(filename_rdg, 'rb'))
result_rdg = loaded_rdg_model.score(X_test_copy, y_test_copy)
print("Ridge Regressor R^2 Score:", result_rdg)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script ml_modeling.ipynb')

