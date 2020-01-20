#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import pickle


# In[3]:


bankset = pd.read_csv('data_banknote_authentication.csv')


# In[4]:


bankset.head()


# In[5]:


from pandas.plotting import scatter_matrix
from matplotlib import pyplot


# In[6]:


scatter_matrix(bankset)
pyplot.show()


# In[7]:


#Classisication - Authentic or Fake


# In[8]:


bankset.isnull().sum()
#no null values


# In[9]:


from sklearn.metrics import accuracy_score


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[11]:


def splitDataset(balance_data):
    X = bankset.iloc[: , :-1]
    Y= bankset.iloc[:,-1]
    X_train, X_validation , Y_train, Y_test = train_test_split(X, Y , test_size = 0.2 , random_state = 2)
    return X,Y, X_train, X_validation ,Y_train, Y_test


# In[12]:


def train_using_gini(X_train , X_validation , Y_train):
    clf_gini = DecisionTreeClassifier(criterion = "gini" , random_state=100 , max_depth=3 , min_samples_leaf=5)
    clf_gini.fit(X_train , Y_train)
    return clf_gini


# In[13]:


def train_using_entropy(X_train , X_validation, Y_train):
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100 , max_depth=3 , min_samples_leaf=5)
    clf_entropy.fit(X_train , Y_train)
    return clf_entropy


# In[14]:


def prediction(X_validation,clf_object):
    Y_pred = clf_object.predict(X_validation)
    return Y_pred


# In[15]:


def calculate_accuracy(Y_test , Y_pred):
    print("Accuracy:" ,accuracy_score(Y_test, Y_pred)*100)


# In[16]:


def main():
    bankset = pd.read_csv('data_banknote_authentication.csv')
    X, Y, X_train, X_validation, Y_train, Y_test = splitDataset(bankset)
    clf_gini = train_using_gini(X_train , X_validation , Y_train)
    clf_entropy = train_using_entropy(X_train, X_validation , Y_train)
    
    #Results using gini index
    y_pred_gini = prediction(X_validation , clf_gini)
    calculate_accuracy(Y_test , y_pred_gini)
    
    #Results using entropy
    y_pred_entropy = prediction(X_validation, clf_entropy)
    calculate_accuracy(Y_test, y_pred_entropy)
    
    if name=="main":
        main()
    


# In[17]:


bankset = pd.read_csv('data_banknote_authentication.csv')
X, Y, X_train, X_validation, Y_train, Y_test = splitDataset(bankset)
clf_gini = train_using_gini(X_train , X_validation , Y_train)
clf_entropy = train_using_entropy(X_train, X_validation , Y_train)


# In[18]:


clf_gini


# In[19]:


y_pred_gini = prediction(X_validation , clf_gini)


# In[20]:


y_pred_gini.shape


# In[21]:


Y_test.shape


# In[22]:


X_validation.shape


# In[23]:


calculate_accuracy(Y_test, y_pred_gini)


# In[24]:


y_pred_ent = prediction(X_validation , clf_entropy)


# In[25]:


calculate_accuracy(Y_test , y_pred_ent)


# In[ ]:





# In[26]:


#Naive Bayes Algorithm


# In[27]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()


# In[29]:


gnb.fit(X_train , Y_train)


y_pred_naivebayes = gnb.predict(X_validation)
# In[30]:

def predict_nb(X_validation):
    y_pred_naivebayes = gnb.predict(X_validation)
    return y_pred_naivebayes


# In[31]:


calculate_accuracy(Y_test , y_pred_naivebayes)


# In[ ]:


# 81% accuracy achieved by Naive Bayes

#Loading model to disk
pickle.dump(gnb , open('model.pkl' , 'wb'))


#Loading model to get the results
pickle.load(open('model.pkl' , 'rb'))