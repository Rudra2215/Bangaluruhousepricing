#!/usr/bin/env python
# coding: utf-8

# # ML AAT "Bangalore House Price Prediction"

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv('Bengaluru_House_Data.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


for column in data.columns:
    print(data[column].value_counts())
    print("*"*20)


# In[7]:


data.isna().sum()


# In[8]:


data.drop(columns=['area_type','availability','society','balcony'],inplace=True)


# In[9]:


data.describe()


# In[10]:


data.info()


# In[11]:


data['location'].value_counts()


# In[12]:


data['location'] = data['location'].fillna('Sarjapur Road')


# In[13]:


data['size'].value_counts()


# In[14]:


data['size'] = data['size'].fillna('2 BHK')


# In[15]:


data['bath']= data['bath'].fillna(data['bath'].median())


# In[16]:


data.info()


# In[17]:


data['bhk']=data['size'].str.split().str.get(0).astype(int)


# In[18]:


data[data.bhk>20]


# In[19]:


data['total_sqft'].unique()


# In[20]:


def convertRange(x):
    
    temp = x.split('-')
    if len(temp)==2:
        return(float(temp[0])+float(temp[1]))/2
    try:
        return float(x)
    except:
        return None


# In[21]:


data['total_sqft']=data['total_sqft'].apply(convertRange)


# In[22]:


data.head()


# # Price per square feet

# In[23]:


data['price_per_sqft']=data['price']*100000/data['total_sqft']


# In[24]:


data['price_per_sqft']


# In[25]:


data.describe()


# In[26]:


data['location'].value_counts()


# In[27]:


data['location']=data['location'].apply(lambda x: x.strip())
location_count= data['location'].value_counts()


# In[28]:


location_count


# In[29]:


location_count_less_10 = location_count[location_count<=10]
location_count_less_10


# In[30]:


data['location']=data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)


# In[31]:


data['location'].value_counts()


# # Outlier detection and removal

# In[32]:


data.describe()


# In[33]:


(data['total_sqft']/data['bhk']).describe()


# In[34]:


data = data[((data['total_sqft']/data['bhk'])>=300)]
data.describe()


# In[35]:


data.shape


# In[36]:


data.price_per_sqft.describe()


# In[37]:


def remove_outlier_sqft(df):
    df_output = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        
        st=np.std(subdf.price_per_sqft)
        
        gen_df = subdf[(subdf.price_per_sqft> (m-st))&(subdf.price_per_sqft<=(m+st))]
        df_output=pd.concat([df_output,gen_df],ignore_index = True)
    return df_output
data = remove_outlier_sqft(data)
data.describe()


# In[38]:


def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
            
        
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
                


# In[39]:


data=bhk_outlier_remover(data)


# In[40]:


data.shape


# In[41]:


data


# In[42]:


data.drop(columns=['size','price_per_sqft'],inplace=True)


# # Cleaned Data 

# In[43]:


data.head()


# In[44]:


data.to_csv("Cleaned_data.csv")


# In[65]:


x=data.drop(columns=['price'])
y=data['price']


# In[98]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


# In[99]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[100]:


print(x_train.shape)
print(x_test.shape)


# # Appliying linear regression

# In[105]:


column_trans = make_column_transformer((OneHotEncoder(sparse_output=False), ['location']), remainder='passthrough')


# In[106]:


scaler = StandardScaler()


# In[108]:


lr = LinearRegression()


# In[92]:


pipe = make_pipeline(column_trans,scaler,lr)


# In[113]:


pipe.fit(x_train,y_train)


# In[114]:


y_pred_lr = pipe.predict(x_test)


# In[116]:


r2_score(y_test,y_pred_lr)


# # APPLYING LASSO

# In[117]:


lasso = Lasso()


# In[118]:


pipe = make_pipeline(column_trans,scaler, lasso)


# In[119]:


pipe.fit(x_train,y_train)


# In[120]:


y_pred_lasso = pipe.predict(x_test)
r2_score(y_test, y_pred_lasso)


# # Apply Ridge Regression

# In[121]:


ridge = Ridge()


# In[122]:


pipe = make_pipeline(column_trans,scaler,ridge)


# In[123]:


pipe.fit(x_train,y_train)


# In[125]:


y_pred_ridge = pipe.predict(x_test)
r2_score(y_test,y_pred_ridge)


# In[128]:


print("No Regularization: ",r2_score(y_test,y_pred_lr))
print("Lasso: ",r2_score(y_test,y_pred_lasso))
print("Ridge: ",r2_score(y_test,y_pred_ridge))


# In[129]:


import pickle


# In[130]:


pickle.dump(pipe, open('RidgeModel.pkl','wb'))


# In[ ]:




