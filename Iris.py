
# coding: utf-8

# In[1]:


# Imports
import numpy as np
import pandas as pd
import pickle
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#import seaborn as sns
#sns.set()
#from sklearn.preprocessing import StandardScaler


# In[2]:


# Load in the data from disk
data = pd.read_csv("iris_train.dat", header=None)
#print(data.shape)
# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]

y = data.iloc[:,cols-1:cols] 
#print(X.shape, y.shape)
# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
#print(np.unique(y))
y = y.flatten()
#print(y.shape)


# In[3]:


data_t = pd.read_csv("iris_test.dat", header=None) 
cols = data_t.shape[1]  
X_test = data_t.iloc[:,0:cols-1]  
y_test = data_t.iloc[:,cols-1:cols] 

X_test = np.array(X_test.values)  
y_test = np.array(y_test.values)
y_test = y_test.flatten()


# In[5]:

#import LR_Regularization_Dropout_Adam 
from LR_Regularization_Dropout_Adam import L_layer_model


# In[6]:


Output_classes = np.unique(y).shape[0]


# In[31]:


learned_parameters = L_layer_model(X, y,Output_classes, layers_dims=[X.shape[1],6,6,np.unique(y).shape[0]], predict_result=False,activation_type="multiclass", reg_type="l2",keep_prob=0.7, mini_batch_size=10, n=1, learning_rate = 0.001,lambd=0.01, num_epochs =500)


# In[32]:


model_to_pickle = "Logisitic_Regression.pkl"
with open(model_to_pickle, 'wb') as file:
    pickle.dump(learned_parameters, file)
