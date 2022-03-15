#!/usr/bin/env python
# coding: utf-8

# # Name:- Parshwa Shah
# # Experiment No.:- 5
# # Roll No.- 34
# # UID:- 2019230071
# # Batch:- B

# # Kaggle:- https://www.kaggle.com/parshwa52

# <h2>Aim:- To measure the performance of the model </h2>

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h3> Import the training dataset </h3>

# In[2]:


dataset = pd.read_csv('./train.csv')


# <h3> Import the testing dataset </h3>

# In[3]:



testdata = pd.read_csv('./test.csv')


# In[4]:


dataset


# <h3> Check the dataset columns </h3>

# In[5]:


dataset.columns


# <h3> Check the columns with null values </h3>

# In[6]:


for col in dataset.columns:
    if(dataset[col].isna().sum()>0):
        #print("Column=",col)
        print(f"Null values in {col} are {dataset[col].isna().sum()}")


# <h3> Check the total no. of rows </h3>

# In[7]:


totalrows=len(dataset)


# In[8]:


totalrows


# <h3> Check data type value counts </h3>

# In[9]:


dataset.dtypes.value_counts()


# <h3> Check for categorical and numerical columns in training data </h3>

# In[10]:


cat_columns=[]
num_columns=[]
for col in dataset.columns.values:
    if dataset[col].dtype=='object':
        cat_columns.append(col)
    else:
        num_columns.append(col)
print(len(cat_columns)," Categorical Columns are \n",cat_columns,'\n')
print(len(num_columns),"Numeric columns are \n",num_columns)

cat_data=dataset[cat_columns]
num_data=dataset[num_columns]


# <h3> Remove numerical columns whose Nan values > 40% and replace columns rest with median values in train data</h3>

# In[11]:


print("Data Size Before Numerical NAN Column(>40%) Removal :",num_data.shape)
for col in num_data.columns.values:
    if (pd.isna(num_data[col]).sum())>0:
        if pd.isna(num_data[col]).sum() > (40/100*len(num_data)):
            print(col,"removed")
            num_data=num_data.drop([col], axis=1)
        else:
            num_data[col]=num_data[col].fillna(num_data[col].median())
print("Data Size After Numerical NAN Column(>40%) Removal :",num_data.shape)


# <h3> Check for categorical and numerical columns in test data </h3>

# In[12]:


#remove columns which have null values > 40%
test_cat_columns=[]
test_num_columns=[]
for col in testdata.columns.values:
    if testdata[col].dtype=='object':
        test_cat_columns.append(col)
    else:
        test_num_columns.append(col)
print(len(test_cat_columns)," Categorical Columns are \n",test_cat_columns,'\n')
print(len(test_num_columns),"Numeric columns are \n",test_num_columns)

test_cat_data=testdata[test_cat_columns]
test_num_data=testdata[test_num_columns]


# <h3> Check for null values in test data </h3>

# In[13]:


test_num_data.isna().sum()


# <h3> Remove numerical columns whose Nan values > 40% and replace columns rest with median values in test data</h3>

# In[14]:


print("Data Size Before Numerical NAN Column(>40%) Removal :",test_num_data.shape)
for col in test_num_data.columns.values:
    if (pd.isna(test_num_data[col]).sum())>0:
        if pd.isna(test_num_data[col]).sum() > (40/100*len(test_num_data)):
            print(col,"removed")
            test_num_data=test_num_data.drop([col], axis=1)
        else:
            test_num_data[col]=test_num_data[col].fillna(test_num_data[col].median())
print("Data Size After Numerical NAN Column(>40%) Removal :",test_num_data.shape)


# In[15]:


test_num_data.isna().sum()


# <h3> Remove categorical columns whose Nan values > 40% and replace columns rest with mode values in test data</h3>

# In[16]:


print("Data Size Before Categorical NAN Column(>40%) Removal :",test_cat_data.shape)
for col in test_cat_data.columns.values:
    if (pd.isna(test_cat_data[col]).sum())>0:
        if pd.isna(test_cat_data[col]).sum() > (40/100*len(test_cat_data)):
            print(col,"removed")
            test_cat_data=test_cat_data.drop([col], axis=1)
        else:
            test_cat_data[col]=test_cat_data[col].fillna(test_cat_data[col].mode()[0])
print("Data Size After Categorical NAN Column(>40%) Removal :",test_cat_data.shape)


# In[17]:


test_cat_data


# In[58]:


import seaborn as sns
sns.scatterplot(x="FullBath",
                    y="SalePrice",
                    data=num_data)


# In[65]:


import seaborn as sns
sns.scatterplot(x="KitchenAbvGr",
                    y="SalePrice",
                    data=num_data)


# In[66]:


import seaborn as sns
sns.scatterplot(x="GarageCars",
                    y="SalePrice",
                    data=num_data)


# 
#     Inference: 
#     1) More bathroom size, more is SalePrice
#     2) KitchenAbvGr is 1, then more is SalePrice
#     3) GarageCars more than 2, more is SalePrice

# <h3> Create a baseline model using numerical columns of train data</h3>

# In[18]:


y = num_data['SalePrice']
X = num_data.drop(['SalePrice'],axis=1)


# In[19]:


X


# In[20]:


y


# <h3> Fit Linear Regression model as baseline model and note R2 score 0.81</h3>

# In[21]:


from sklearn.linear_model import LinearRegression
basereg = LinearRegression().fit(X, y)
baseypred = basereg.predict(test_num_data)
basereg.score(X, y)


# Referring Regression ROC AUC Score from:
# https://towardsdatascience.com/how-to-calculate-roc-auc-score-for-regression-models-c0be4fdf76bb

# In[22]:


def regression_roc_auc_score(y_true, y_pred, num_rounds = 10):
  """
  Computes Regression-ROC-AUC-score.
  
  Parameters:
  ----------
  y_true: array-like of shape (n_samples,). Binary or continuous target variable.
  y_pred: array-like of shape (n_samples,). Target scores.
  num_rounds: int or string. If integer, number of random pairs of observations. 
              If string, 'exact', all possible pairs of observations will be evaluated.
  
  Returns:
  -------
  rroc: float. Regression-ROC-AUC-score.
  """
  
  import numpy as np
    
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  num_pairs = 0
  num_same_sign = 0
  
  for i, j in _yield_pairs(y_true, num_rounds):
    diff_true = y_true[i] - y_true[j]
    diff_score = y_pred[i] - y_pred[j]
    if diff_true * diff_score > 0:
      num_same_sign += 1
    elif diff_score == 0:
      num_same_sign += .5
    num_pairs += 1
      
  return num_same_sign / num_pairs


def _yield_pairs(y_true, num_rounds):
  """
  Returns pairs of valid indices. Indices must belong to observations having different values.
  
  Parameters:
  ----------
  y_true: array-like of shape (n_samples,). Binary or continuous target variable.
  num_rounds: int or string. If integer, number of random pairs of observations to return. 
              If string, 'exact', all possible pairs of observations will be returned.
  
  Yields:
  -------
  i, j: tuple of int of shape (2,). Indices referred to a pair of samples.
  
  """
  import numpy as np
  
  if num_rounds == 'exact':
    for i in range(len(y_true)):
      for j in np.where((y_true != y_true[i]) & (np.arange(len(y_true)) > i))[0]:
        yield i, j     
  else:
    for r in range(num_rounds):
      i = np.random.choice(range(len(y_true)))
      j = np.random.choice(np.where(y_true != y_true[i])[0])
      yield i, j


# In[23]:


from sklearn.metrics import roc_auc_score
print("ROC AUC Score=",regression_roc_auc_score(y.to_numpy(),baseypred,10))


# <h3> Fit Random Forest Regression model as baseline model and note R2 score 0.70</h3>

# In[24]:


#baseline model on numerical data
from sklearn.ensemble import RandomForestRegressor
baserf = RandomForestRegressor(max_depth=2, random_state=0)
baserf.fit(X, y)
baserf.score(X,y)


# <h3> Remove categorical columns whose Nan values > 40% and replace columns rest with mode values in train data</h3>

# In[25]:


print("Data Size Before Categorical NAN Column(>40%) Removal :",cat_data.shape)
for col in cat_data.columns.values:
    if (pd.isna(cat_data[col]).sum())>0:
        if pd.isna(cat_data[col]).sum() > (40/100*len(cat_data)):
            print(col,"removed")
            cat_data=cat_data.drop([col], axis=1)
        else:
            cat_data[col]=cat_data[col].fillna(cat_data[col].mode()[0])
print("Data Size After Categorical NAN Column(>40%) Removal :",cat_data.shape)


# <h3> Check null values in numeric and categorical data</h3>

# In[26]:


num_data


# In[27]:


num_data.isna().sum()


# In[28]:


cat_data


# In[29]:


cat_data.isna().sum()


# <h3> Remove unnecessary columns from numeric data in train set</h3>

# In[30]:


num_data=num_data.drop(['Id'], axis = 1)


# In[31]:


num_data


# <h3> Derive newness column from yrsold and yearremodadd columns</h3>

# In[32]:


num_data['newness'] = num_data['YrSold'] - num_data['YearRemodAdd']


# In[33]:


num_data


# <h3> Remove year based columns from train data</h3>

# In[34]:


num_data=num_data.drop(['YearBuilt','YearRemodAdd','YrSold'], axis = 1)


# In[35]:


num_data


# <h3> Label encode categorical data in train set</h3>

# In[36]:


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

for col in cat_data.columns:
    label_encoder.fit(cat_data[col])
    cat_data[col]= label_encoder.transform(cat_data[col])
    test_cat_data[col] = label_encoder.transform(test_cat_data[col])


# <h3> Create final train data containing numeric and categorical data</h3>

# In[37]:


finaltraindata=pd.concat([num_data,cat_data],axis=1)
finaltraindata


# <h3> Keep train data ready for model</h3>

# In[38]:


yfinal = finaltraindata['SalePrice']
Xfinal = finaltraindata.drop(['SalePrice'],axis=1)


# In[39]:


Xfinal


# In[40]:


yfinal


# <h3> Check test numeric and categorical data</h3>

# In[41]:


test_num_data


# In[42]:


test_cat_data


# <h3> Apply same preprocessing on test data as train data</h3>

# In[43]:


testid = test_num_data['Id']
test_num_data=test_num_data.drop(['Id'], axis = 1)


# In[44]:


testid


# In[45]:


test_num_data['newness'] = test_num_data['YrSold'] - test_num_data['YearRemodAdd']


# In[46]:


test_num_data=test_num_data.drop(['YearBuilt','YearRemodAdd','YrSold'], axis = 1)


# In[47]:


finaltestdata=pd.concat([test_num_data,test_cat_data],axis=1)
finaltestdata


# <h3> Create Linear Regression as Model 1</h3>

# In[48]:


from sklearn.linear_model import LinearRegression
model1reg = LinearRegression().fit(Xfinal, yfinal)
model1pred = model1reg.predict(finaltestdata)
model1reg.score(Xfinal, yfinal)


# <h3> Get results on test set for Linear Regression model and store results in csv file</h3>

# In[49]:


model1pred


# In[50]:


iddf = pd.DataFrame(testid, columns = ['Id'])


# In[51]:


def convert_to_csv(modelpred,iddf,idn):
    model1df = pd.DataFrame(modelpred, columns = ['SalePrice'])
    frames = [iddf, model1df]
    result = pd.concat(frames,axis=1)
    result.to_csv(f'ytestres{idn}.csv',index=False)


# In[52]:


convert_to_csv(model1pred,iddf,0)


# <h3> Create Decision Tree Regression as Model 2</h3>

# In[53]:


from sklearn.tree import DecisionTreeRegressor
model2rf = DecisionTreeRegressor()
model2rf.fit(Xfinal, yfinal)
model2pred = model2rf.predict(finaltestdata)


# In[54]:


convert_to_csv(model2pred,iddf,1)


# Conclusion:- Hence, in this experiment, I participated in Kaggle competition. I preprocessed the data and first created baseline model. Then, I encoded data and added some columns and finally after cleaning, I used two models, Linear Regression and Decision Tree Regression and submitted the results.
