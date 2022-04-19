#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# para evitarmos a exibição dos dados em notacao científica
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[ ]:


from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from numpy import mean
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


# In[ ]:


#Reload dataset cleaned
dataset = pd.read_csv("~/biomarcprogml_ov/results/Leukemia/version2022/leukemia247.csv", index_col=0)
df = pd.DataFrame(dataset)
df.shape


# In[ ]:

#filter top 20 genes of best model (LightGBM)

print(df.head())
df = df[['Status','Leukemia', 'Age_group', 'SKIDA1', "LAMP5", "HOXA9", "SOCS2", "CLEC2B", "PPP1R27", "CPA6", "NEDD4", "SERINC2", "SLC35G1", "TRPM4", "MEIS1", "FEZ1", "MYO5C", "ZNF254", "GOLGA8I", "MYO6", "VAT1L", "HTR1F", "MEF2C"]]
print(df.head())

# In[ ]:


df.Status.value_counts()


# In[ ]:


print(df.Leukemia.value_counts())


# In[ ]:


# Remove the column Age_group
df = df.drop(['Age_group'],axis = 1)
# create new variables with the dummies function with the categorical variable leukemia
# (e.g: 'Leukemia_ALAL', 'Leukemia_AML', 'Leukemia_B-ALL', 'Leukemia_T-ALL')
df = pd.get_dummies(df, columns=['Leukemia'], dtype=int)

print(df.head())

# #  Model Development
# 
# ### I define the class to be predicted (Y)
#  

# In[ ]:


def joinCategories(row):
    if row['Status']== 'KMT2A-r'  :
        val = 1
    else:
        val = 0
    return val


# In[ ]:


df['Status'] = df.apply(joinCategories, axis=1)


# In[ ]:


df.Status.value_counts()


# In[ ]:


df.head()
print(df.shape)

# In[ ]:


X=df.drop(['Status'],axis = 1)
y=df[['Status']] 



# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)


# In[ ]:

from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None

# Normalize the data excluding the categorical variables
continuas_cols = X.iloc[:, ~X.columns.isin(['Leukemia_ALAL', 'Leukemia_AML', 'Leukemia_B-ALL',
       'Leukemia_T-ALL'])]
categoricas_cols = X.iloc[:, X.columns.isin(['Leukemia_ALAL', 'Leukemia_AML', 'Leukemia_B-ALL',
       'Leukemia_T-ALL'])]


sc = StandardScaler()
X_train[continuas_cols.columns] = sc.fit_transform(X_train[continuas_cols.columns])
X_test[continuas_cols.columns] = sc.transform(X_test[continuas_cols.columns])



# # Resample

# In[ ]:

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
np.random.seed(42)
resample = SMOTEENN()
# define pipeline
pipeline = Pipeline(steps=[('r', resample)])
#pipeline = Pipeline(steps=steps)
# transform the dataset
X_trains, y_trains = pipeline.fit_resample(X_train, y_train)


# In[ ]:

#save train and test datasets
X_trains.to_csv("X_trains.csv")
y_trains.to_csv("y_trains.csv")
X_test.to_csv("X_test.csv")
y_test.to_csv("y_test.csv")


# In[ ]:


y_trains.Status.value_counts()


# In[ ]:


X_trains.head()


# In[ ]:


X_trains.shape
print('Predictor rows and columns in training:',X_trains.shape)

X_test.shape
print('Predictor rows and columns in test:',X_test.shape)

y_trains.shape
print('Outcome rows and column on training:',y_trains.shape)

y_test.shape
print('Outcome rows and column on test:',y_test.shape)


# In[ ]:


np.random.seed(42)


# # Random Forest
# 
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# define pipeline

rf= RandomForestClassifier(random_state=42)
#Kfold Cross-validation
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 5)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 20, num = 5)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the param grid

grid_params = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


metricas = {'roc_auc','recall', 'f1','accuracy', 'precision'}


grid_RF=GridSearchCV(rf,
                    param_grid=grid_params,
                    scoring=metricas,
                    verbose=1,
                    refit='roc_auc',
                    cv=cv,
                    n_jobs = -1,
                    return_train_score = False)

grid_RF.fit(X_trains, y_trains.values.ravel())


# In[ ]:


print('Best score: %s' % grid_RF.best_score_)
print('Best params: %s' % grid_RF.best_params_)


# In[ ]:


import pickle
best_rf = grid_RF.best_estimator_

#save model
grid = open('rf.pkl', 'wb')
pickle.dump(best_rf, grid)
grid.close()


# In[ ]:


np.random.seed(42)
rf.fit(X_trains,y_trains.values.ravel())



# In[ ]:


from sklearn.metrics import roc_auc_score
y_pred = best_rf.fit(X_trains,y_trains.values.ravel()).predict(X_test)
print(roc_auc_score(y_test, y_pred))


# In[ ]:


from sklearn.dummy import DummyClassifier
# no skill model, stratified random class predictions
modeld = DummyClassifier(strategy='stratified', random_state=42)
modeld.fit(X_trains, y_trains.values.ravel())
yhatd = modeld.predict_proba(X_test)
pos_probs = yhatd[:, 1]
# calculate roc auc
roc_auc = roc_auc_score(y_test, pos_probs)
print('No Skill ROC AUC %.3f' % roc_auc)
# skilled model
best_rf.fit(X_trains,  y_trains.values.ravel())
yhat = best_rf.predict_proba(X_test)
pos_probs = yhat[:,1]
# calculate roc auc
roc_auc = roc_auc_score(y_test.values.ravel(), pos_probs)
print('Random Forest ROC AUC %.3f' % roc_auc)


# In[ ]:


from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)


# # LightGBM

# In[ ]:


import lightgbm as lgb

classifier = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)
rs_parameters = {
    'learning_rate': [0.005,0.01,0.001,0.05],
    'n_estimators': [20,40,60,80,100],
   'num_leaves': [6,8,12,16]
   }

metricas = {'roc_auc','recall', 'f1','accuracy', 'precision'}


gridL = GridSearchCV(classifier,
                         param_grid=rs_parameters,
                         cv=cv,
                         scoring=metricas,
                         refit='roc_auc',
                         return_train_score=False,
                         n_jobs=-1,
                         verbose=True)

gridL.fit(X_trains, y_trains.values.ravel())


# In[ ]:


print('Best score: %s' % gridL.best_score_)
print('Best params: %s' % gridL.best_params_)


# In[ ]:


import pickle
gbm = gridL.best_estimator_

#save model
grid = open('gbm.pkl', 'wb')
pickle.dump(gbm, grid)
grid.close()


# In[ ]:


np.random.seed(42)
gbm.fit(X_trains,y_trains.values.ravel())




# In[ ]:


from sklearn.metrics import roc_auc_score
y_pred = gbm.fit(X_trains,y_trains.values.ravel()).predict(X_test)
print(roc_auc_score(y_test, y_pred))


# In[ ]:


from sklearn.dummy import DummyClassifier
# no skill model, stratified random class predictions
modeld = DummyClassifier(strategy='stratified', random_state=42)
modeld.fit(X_trains, y_trains.values.ravel())
yhatd = modeld.predict_proba(X_test)
pos_probs = yhatd[:, 1]
# calculate roc auc
roc_auc = roc_auc_score(y_test, pos_probs)
print('No Skill ROC AUC %.3f' % roc_auc)
# skilled model
gbm.fit(X_trains,  y_trains.values.ravel())
yhat = gbm.predict_proba(X_test)
pos_probs = yhat[:,1]
# calculate roc auc
roc_auc = roc_auc_score(y_test.values.ravel(), pos_probs)
print('Light GBM ROC AUC %.3f' % roc_auc)


# In[ ]:


from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)
