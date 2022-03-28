#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def get_data(name):
    train_name = name + '_train_new.csv'
    if name == 'tag':
        test_name = '提交样例.csv'
    else:
        test_name = name + '_round1_new.csv'
    train_data_path = os.path.join(path, train_name)
    test_data_path = os.path.join(path, test_name)
    train_data = pd.read_csv('../' + train_data_path)
    test_data = pd.read_csv('../' + test_data_path)
    train_data = pd.concat([train_data, test_data], axis=0, ignore_index=True, sort=False)
    if name == 'tag':
        train_data.loc[train_data['Tag'] == 0.5, 'Tag'] = -1
        train_data['Tag'] = train_data['Tag'].astype(int)
    return train_data


# In[3]:


path = '../01-data'

operation_data = get_data(name='operation')
transaction_data = get_data(name='transaction')
tag = get_data(name='tag')

print(operation_data.shape)
print(transaction_data.shape)
print(tag.shape)


# In[4]:


op_data = pd.merge(operation_data, tag, on='UID', how='left')
tr_data = pd.merge(transaction_data, tag, on='UID', how='left')

print(op_data.shape)
print(tr_data.shape)


# In[5]:


tag['Tag'].value_counts().plot.pie(autopct = '%1.2f%%')


# # Operation

# In[6]:


op_data.head()


# In[7]:


op_data.loc[op_data['device_code1'].isnull(),'device_code1'] = op_data[op_data['device_code1'].isnull()]['device_code2']
op_data.loc[op_data['device_code1'].isnull(),'device_code1'] = op_data[op_data['device_code1'].isnull()]['device_code3']
op_data = op_data.drop(['device_code2','device_code3'],axis=1)


# In[8]:


op_data.loc[op_data['ip1'].isnull(),'ip1'] = op_data[op_data['ip1'].isnull()]['ip2']
op_data = op_data.drop(['ip2'],axis=1)


# In[9]:


op_data.loc[op_data['ip1_sub'].isnull(),'ip1_sub'] = op_data[op_data['ip1_sub'].isnull()]['ip2_sub']
op_data = op_data.drop(['ip2_sub'],axis=1)


# In[10]:


op_data.loc[op_data['mac2'].isnull(),'mac2'] = op_data[op_data['mac2'].isnull()]['mac1']
op_data = op_data.drop(['mac1'],axis=1)


# In[11]:


op_data.head()


# In[12]:


cols = ['mode', 'os',  'device2',  'geo_code', 'ip1_sub','version', 'mac2','device1', 'device_code1', 'ip1', 'wifi']
for col in cols:
    op_data[col] = pd.factorize(op_data[col])[0]
op_data.head()


# In[13]:


op_data['hour'] = op_data.time.str[0:2].astype(int)

op_data['time'] = pd.to_datetime(op_data['time'])
op_data['time'] = op_data['time'].apply(lambda x:(x-datetime.now()).seconds/3600)

op_data.head()


# In[14]:


op_data_na = op_data.replace(-1,np.nan)
op_data_na = (op_data_na.isnull().sum()/len(op_data_na)) * 100
op_data_na = op_data_na.drop(op_data_na[op_data_na == 0].index).sort_values(ascending = True)
missing_op_data = pd.DataFrame({'missing ratio': op_data_na})
print("the number of missing item: {}".format(len(missing_op_data)))
missing_op_data


# In[15]:


op_data_temp = op_data
op_data_temp.shape


# In[16]:


# op_data = op_data_temp


# In[17]:


op_data = op_data.merge(op_data.groupby(['hour'], as_index=False)['success'].agg({'hour'+'_success_count':'count'}),on='hour',how='left')
op_data = op_data.merge(op_data.groupby(['hour'], as_index=False)['success'].agg({'hour'+'_success_sum':'sum'}),on='hour',how='left')
op_data['hour_rate'] = op_data['hour_success_sum']/op_data['hour_success_count']
op_data.shape


# In[18]:


op_data = op_data.merge(op_data.groupby(['mode'], as_index=False)['success'].agg({'mode'+'_success_count':'count'}),on='mode',how='left')
op_data = op_data.merge(op_data.groupby(['mode'], as_index=False)['success'].agg({'mode'+'_success_sum':'sum'}),on='mode',how='left')
op_data['mode_rate'] = op_data['mode_success_sum']/op_data['mode_success_count']
op_data.shape


# In[19]:


op_data = op_data.fillna(-1)


# In[20]:


op_data.head()


# In[21]:


#统计特征
def get_category_uid(data_set,feature):
    
    data_set = data_set.merge(data_set.groupby([feature], as_index=False)['UID'].agg({feature+'_nunique':'nunique'}),on=feature,how='left')
    
    data_set = data_set.merge(data_set.groupby([feature], as_index=False)['UID'].agg({feature+'_count':'count'}),on=feature,how='left')

    return data_set


# In[22]:


#day
print('day...')
op_data = get_category_uid(op_data, 'day')
#hour
print('hour...')
op_data = get_category_uid(op_data, 'hour')
##mode
print('mode...')
op_data = get_category_uid(op_data, 'mode')
#success
print('success...')
op_data = get_category_uid(op_data, 'success')
#os
print('os...')
op_data = get_category_uid(op_data, 'os')
#version
print('version...')
op_data = get_category_uid(op_data, 'version')
#device1
print('device1...')
op_data = get_category_uid(op_data, 'device1')
#device2
print('device2...')
op_data = get_category_uid(op_data, 'device2')
#mac2
print('mac2...')
op_data = get_category_uid(op_data, 'mac2')
#geo_code
print('geo_code...')
op_data = get_category_uid(op_data, 'geo_code')
#ip1_sub
print('ip1_sub...')
op_data = get_category_uid(op_data, 'ip1_sub')
#ip1
print('ip1...')
op_data = get_category_uid(op_data, 'ip1')
#wifi
print('wifi...')
op_data = get_category_uid(op_data, 'wifi')
#device_code
print('device_code1...')
op_data = get_category_uid(op_data, 'device_code1')

print(op_data.shape)


# In[23]:


op_data.head()


# In[24]:


n_train_x = 1460843

train_x1 = op_data[:n_train_x]
test_x1 = op_data[n_train_x:]
print(train_x1.shape)
print(test_x1.shape)


# # Transaction

# In[25]:


# tr_data = pd.merge(transaction_data, tag, on='UID', how='left')


# In[26]:


tr_data.head()


# In[27]:


tr_data.loc[tr_data['device_code1'].isnull(),'device_code1'] = tr_data[tr_data['device_code1'].isnull()]['device_code2']
tr_data.loc[tr_data['device_code1'].isnull(),'device_code1'] = tr_data[tr_data['device_code1'].isnull()]['device_code3']
tr_data = tr_data.drop(['device_code2','device_code3'],axis=1)


# In[28]:


tr_data = tr_data.drop(['acc_id2'],axis=1)


# In[29]:


cols_f = ['amt_src1', 'trans_type1','amt_src2','geo_code','ip1_sub','mac1','merchant','acc_id3','channel','market_type','trans_type2','code1','code2','market_code','ip1','device1','device_code1','acc_id1','device2']
for col in cols_f:
    tr_data[col] = pd.factorize(tr_data[col])[0]
tr_data.head()


# In[30]:


tr_data['hour'] = tr_data.time.str[0:2].astype(int)

tr_data['time'] = pd.to_datetime(tr_data['time'])
tr_data['time'] = tr_data['time'].apply(lambda x:(x-datetime.now()).seconds/3600)

tr_data.head()


# In[31]:


tr_data_na = tr_data.replace(-1,np.nan)
tr_data_na = (tr_data_na.isnull().sum()/len(tr_data_na)) * 100
tr_data_na = tr_data_na.drop(tr_data_na[tr_data_na == 0].index).sort_values(ascending = True)
missing_tr_data = pd.DataFrame({'missing ratio': tr_data_na})
print("the number of missing item: {}".format(len(missing_tr_data)))
missing_tr_data


# In[32]:


tr_data = tr_data.fillna(-1)


# In[33]:


tr_data_temp = tr_data
tr_data_temp.shape


# In[34]:


tr_data = tr_data.merge(tr_data.groupby(['channel'], as_index=False)['trans_amt'].agg({'channel'+'trans_sum':'sum'}),on='channel',how='left')


# In[35]:


#day
print('day...')
tr_data = get_category_uid(tr_data, 'day')
#hour
print('hour...')
tr_data = get_category_uid(tr_data, 'hour')
#channel
print('channel...')
tr_data = get_category_uid(tr_data, 'channel')
#merchant
print('merchant...')
tr_data = get_category_uid(tr_data, 'merchant')
#amt_src1
print('amt_src1...')
tr_data = get_category_uid(tr_data, 'amt_src1')
#amt_src2
print('amt_src2...')
tr_data = get_category_uid(tr_data, 'amt_src2')
# #code1
# print('code1...')
# tr_data = get_category_uid(tr_data, 'code1')
# #code2
# print('code2...')
# tr_data = get_category_uid(tr_data, 'code2')
#device1
print('device1...')
tr_data = get_category_uid(tr_data, 'device1')
#device2
print('device2...')
tr_data = get_category_uid(tr_data, 'device2')
#mac1
print('mac1...')
tr_data = get_category_uid(tr_data, 'mac1')
#geo_code
print('geo_code...')
tr_data = get_category_uid(tr_data, 'geo_code')
#ip1_sub
print('ip1_sub...')
tr_data = get_category_uid(tr_data, 'ip1_sub')
#ip1
print('ip1...')
tr_data = get_category_uid(tr_data, 'ip1')
#acc_id1
print('acc_id1...')
tr_data = get_category_uid(tr_data, 'acc_id1')
# #acc_id3
# print('acc_id3...')
# tr_data = get_category_uid(tr_data, 'acc_id3')
#device_code1
print('device_code1...')
tr_data = get_category_uid(tr_data, 'device_code1')
#trans_type1
print('trans_type1...')
tr_data = get_category_uid(tr_data, 'trans_type1')
#trans_type2
print('trans_type2...')
tr_data = get_category_uid(tr_data, 'trans_type2')
# #market_type
# print('market_type...')
# tr_data = get_category_uid(tr_data, 'market_type')
# #market_code
# print('market_code...')
# tr_data = get_category_uid(tr_data, 'market_code')

print(tr_data.shape)


# In[36]:


n_train_x2 = 264654

train_x2 = tr_data[:n_train_x2]
test_x2 = tr_data[n_train_x2:]
print(train_x2.shape)
print(test_x2.shape)


# # Merge

# In[37]:


#类别型变量
def get_category_feature(data_set,feature,train_y):
    
    train_y = train_y.merge(data_set.groupby(['UID'])[feature].agg(lambda x: x.value_counts().index[0]),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].max(),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].nunique(),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].min(),on='UID',how='left')
    
    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].sum(),on='UID',how='left')

    return train_y


# In[38]:


#数值型变量
def get_numerical_feature(data_set,feature,train_y):
    
    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].var(),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].mad(),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].sum(),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].mean(),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].min(),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].median(),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].max(),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].std(),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'], as_index=False)[feature].skew(),on='UID',how='left')

    train_y = train_y.merge(data_set.groupby(['UID'])[feature].agg(lambda x: x.value_counts().index[0]),on='UID',how='left')

    return train_y


# In[39]:


#op_data merge
train_y = pd.read_csv('../../01-data/tag_train_new.csv')


# In[40]:


train_x1.columns


# In[41]:


#day
print('day...')
op_train = train_y.merge(train_x1.groupby(['UID'], as_index=False)['day'].agg({'count':'count'}),on='UID',how='left')
op_train = get_category_feature(train_x1, 'day', op_train)
##mode
print('mode...')
op_train = get_category_feature(train_x1, 'mode', op_train)
#success
print('success...')
op_train = get_category_feature(train_x1, 'success', op_train)
#os
print('os...')
op_train = get_category_feature(train_x1, 'os', op_train)
#version
print('version...')
op_train = get_category_feature(train_x1, 'version', op_train)
#device1
print('device1...')
op_train = get_category_feature(train_x1, 'device1', op_train)
#device2
print('device2...')
op_train = get_category_feature(train_x1, 'device2', op_train)
#device_code1
print('device_code1...')
op_train = get_category_feature(train_x1, 'device_code1', op_train)
#mac2
print('mac2...')
op_train = get_category_feature(train_x1, 'mac2', op_train)
#geo_code
print('geo_code...')
op_train = get_category_feature(train_x1, 'geo_code', op_train)
#ip1_sub
print('ip1_sub...')
op_train = get_category_feature(train_x1, 'ip1_sub', op_train)
#ip1
print('ip1...')
op_train = get_category_feature(train_x1, 'ip1', op_train)
#wifi
print('wifi...')
op_train = get_category_feature(train_x1, 'wifi', op_train)
#time
print('time...')
op_train = get_numerical_feature(train_x1, 'time', op_train)
print(op_train.shape)


# In[42]:


cols = ['hour_rate', 'mode_rate']
for col in cols:
    print(col)
    op_train = get_numerical_feature(train_x1, col, op_train)
print(op_train.shape)


# In[43]:


cols = ['hour_success_count', 'hour_success_sum', 'mode_success_count', 'mode_success_sum',  'day_nunique',
       'day_count', 'hour_nunique', 'hour_count', 'mode_nunique', 'mode_count',
       'success_nunique', 'success_count', 'os_nunique', 'os_count',
       'version_nunique', 'version_count', 'device1_nunique', 'device1_count',
       'device2_nunique', 'device2_count', 'mac2_nunique', 'mac2_count',
       'geo_code_nunique', 'geo_code_count', 'ip1_sub_nunique',
       'ip1_sub_count', 'ip1_nunique', 'ip1_count', 'wifi_nunique',
       'wifi_count', 'device_code1_nunique', 'device_code1_count']
for col in cols:
    print(col)
    op_train = get_category_feature(train_x1, col, op_train)
print(op_train.shape)


# In[44]:


op_train.head()


# In[46]:


test_y = pd.read_csv('../../01-data/提交样例.csv')


# In[47]:


#day
print('day...')
op_test = test_y.merge(test_x1.groupby(['UID'], as_index=False)['day'].agg({'count':'count'}),on='UID',how='left')
op_test = get_category_feature(test_x1, 'day', op_test)
##mode
print('mode...')
op_test = get_category_feature(test_x1, 'mode', op_test)
#success
print('success...')
op_test = get_category_feature(test_x1, 'success', op_test)
#os
print('os...')
op_test = get_category_feature(test_x1, 'os', op_test)
#version
print('version...')
op_test = get_category_feature(test_x1, 'version', op_test)
#device1
print('device1...')
op_test = get_category_feature(test_x1, 'device1', op_test)
#device2
print('device2...')
op_test = get_category_feature(test_x1, 'device2', op_test)
#device_code1
print('device_code1...')
op_test = get_category_feature(test_x1, 'device_code1', op_test)
#mac2
print('mac2...')
op_test = get_category_feature(test_x1, 'mac2', op_test)
#geo_code
print('geo_code...')
op_test = get_category_feature(test_x1, 'geo_code', op_test)
#ip1
print('ip1...')
op_test = get_category_feature(test_x1, 'ip1', op_test)
#wifi
print('wifi...')
op_test = get_category_feature(test_x1, 'wifi', op_test)
#ip1_sub
print('ip1_sub...')
op_test = get_category_feature(test_x1, 'ip1_sub', op_test)
#time
print('time...')
op_test = get_numerical_feature(test_x1, 'time', op_test)
print(op_test.shape)


# In[48]:


cols = ['hour_rate','mode_rate']
for col in cols:
    print(col)
    op_test = get_numerical_feature(test_x1, col, op_test)
print(op_test.shape)


# In[49]:


cols = ['hour_success_count', 'hour_success_sum', 'mode_success_count', 'mode_success_sum',  'day_nunique',
       'day_count', 'hour_nunique', 'hour_count', 'mode_nunique', 'mode_count',
       'success_nunique', 'success_count', 'os_nunique', 'os_count',
       'version_nunique', 'version_count', 'device1_nunique', 'device1_count',
       'device2_nunique', 'device2_count', 'mac2_nunique', 'mac2_count',
       'geo_code_nunique', 'geo_code_count', 'ip1_sub_nunique',
       'ip1_sub_count', 'ip1_nunique', 'ip1_count', 'wifi_nunique',
       'wifi_count', 'device_code1_nunique', 'device_code1_count']
for col in cols:
    print(col)
    op_test = get_category_feature(test_x1, col, op_test)
print(op_test.shape)


# In[50]:


op_test.head()


# In[51]:


aa = op_train
aaa = aa.iloc[:,2:]
qq = np.array(aaa)
pp = pd.DataFrame(qq)
aaaa = aa.iloc[:,0:2]
aaaaa = pd.concat([aaaa,pp],axis=1)
cc = 0
for col in aaaaa.columns[2:]:
    aaaaa.rename(columns= {col: 'feature' + '_'+ str(cc)}, inplace=True)
    cc = cc + 1

bb = op_test
bbb = bb.iloc[:,2:]
qqq = np.array(bbb)
ppp = pd.DataFrame(qqq)
bbbb = bb.iloc[:,0:2]
bbbbb = pd.concat([bbbb,ppp],axis=1)
ccc = 0
for col in bbbbb.columns[2:]:
    bbbbb.rename(columns= {col: 'feature' + '_'+ str(ccc)}, inplace=True)
    ccc = ccc + 1  

op_train1 = aaaaa
op_test1 = bbbbb

op_test1.Tag = -1

all_data = pd.concat([op_train1, op_test1])


# In[52]:


all_data.head()


# In[53]:


op_train2 = all_data[all_data['Tag']!=-1]
op_test2 = all_data[all_data['Tag']==-1]
print(op_train2.shape)
print(op_test2.shape)


# In[54]:


train_x2.columns


# In[ ]:


#day
print('day...')
op_train3 = op_train2.merge(train_x2.groupby(['UID'], as_index=False)['day'].agg({'count':'count'}),on='UID',how='left')
op_train3 = get_category_feature(train_x2, 'day', op_train3)
#channel
print('channel...')
op_train3 = get_category_feature(train_x2, 'channel', op_train3)
# trans_amt
print('trans_amt...')
op_train3 = get_numerical_feature(train_x2, 'trans_amt', op_train3)
#merchant
print('merchant...')
op_train3 = get_category_feature(train_x2, 'merchant', op_train3)
#trans_type1
print('trans_type1...')
op_train3 = get_category_feature(train_x2, 'trans_type1', op_train3)
#mac1
print('mac1...')
op_train3 = get_category_feature(train_x2, 'mac1', op_train3)
#bal
print('bal...')
op_train3 = get_numerical_feature(train_x2, 'bal', op_train3)
#amt_src2
print('amt_src2...')
op_train3 = get_category_feature(train_x2, 'amt_src2', op_train3)
#acc_id1
print('acc_id1...')
op_train3 = get_category_feature(train_x2, 'acc_id1', op_train3)  
#geo_code
print('geo_code...')
op_train3 = get_category_feature(train_x2, 'geo_code', op_train3)
#trans_type2
print('trans_type2...')
op_train3 = get_category_feature(train_x2, 'trans_type2', op_train3)
#channeltrans_sum
print('channeltrans_sum...')
op_train3 = get_category_feature(train_x2, 'channeltrans_sum', op_train3)
#device_code1
print('device_code1...')
op_train3 = get_category_feature(train_x2, 'device_code1', op_train3)
#ip1_sub
print('ip1_sub...')
op_train3 = get_category_feature(train_x2, 'ip1_sub', op_train3)
#time
print('time...')
op_train3 = get_numerical_feature(train_x2, 'time', op_train3)
#market_type
# print('market_type...')
# op_train3 = get_category_feature(train_x2, 'market_type', op_train3)
#market_code
# print('market_code...')
# op_train3 = get_category_feature(train_x2, 'market_code', op_train3)

print(op_train3.shape)


# In[ ]:


cols = ['day_nunique', 'day_count', 'hour_nunique',
       'hour_count', 'channel_nunique', 'channel_count', 'merchant_nunique',
       'merchant_count', 'amt_src1_nunique', 'amt_src1_count',
       'amt_src2_nunique', 'amt_src2_count', 'device1_nunique',
       'device1_count', 'device2_nunique', 'device2_count', 'mac1_nunique',
       'mac1_count', 'geo_code_nunique', 'geo_code_count', 'ip1_sub_nunique',
       'ip1_sub_count', 'ip1_nunique', 'ip1_count', 'acc_id1_nunique',
       'acc_id1_count', 'device_code1_nunique', 'device_code1_count',
       'trans_type1_nunique', 'trans_type1_count', 'trans_type2_nunique',
       'trans_type2_count']
for col in cols:
    print(col)
    op_train3 = get_category_feature(train_x2, col, op_train3)
print(op_train3.shape)


# In[ ]:


op_train3.shape


# In[ ]:


#day
print('day...') 
op_test3 = op_test2.merge(test_x2.groupby(['UID'], as_index=False)['day'].agg({'count':'count'}),on='UID',how='left')
op_test3 = get_category_feature(test_x2, 'day', op_test3)
#channel
print('channel...')
op_test3 = get_category_feature(test_x2, 'channel', op_test3)
#trans_amt
print('trans_amt...')
op_test3 = get_numerical_feature(test_x2, 'trans_amt', op_test3)
#merchant
print('merchant...')
op_test3 = get_category_feature(test_x2, 'merchant', op_test3)
#trans_type1
print('trans_type1...')
op_test3 = get_category_feature(test_x2, 'trans_type1', op_test3)
#mac1
print('mac1...')
op_test3 = get_category_feature(test_x2, 'mac1', op_test3)
#bal
print('bal...')
op_test3 = get_numerical_feature(test_x2, 'bal', op_test3)
#amt_src2
print('amt_src2...')
op_test3 = get_category_feature(test_x2, 'amt_src2', op_test3)
#acc_id1
print('acc_id1...')
op_test3 = get_category_feature(test_x2, 'acc_id1', op_test3)
#geo_code
print('geo_code...')
op_test3 = get_category_feature(test_x2, 'geo_code', op_test3)
#trans_type2
print('trans_type2...')
op_test3 = get_category_feature(test_x2, 'trans_type2', op_test3)
#channeltrans_sum
print('channeltrans_sum...')
op_test3 = get_category_feature(test_x2, 'channeltrans_sum', op_test3)
#device_code1
print('device_code1...')
op_test3 = get_category_feature(test_x2, 'device_code1', op_test3) 
#ip1_sub
print('ip1_sub...')
op_test3 = get_category_feature(test_x2, 'ip1_sub', op_test3)
#time
print('time...')
op_test3 = get_numerical_feature(test_x2, 'time', op_test3)
#market_type
# print('market_type...')
# op_test3 = get_category_feature(test_x2, 'market_type', op_test3)
#market_code
# print('market_code...')
# op_test3 = get_category_feature(test_x2, 'market_code', op_test3)


print(op_test3.shape)


# In[ ]:


cols = ['day_nunique', 'day_count', 'hour_nunique',
       'hour_count', 'channel_nunique', 'channel_count', 'merchant_nunique',
       'merchant_count', 'amt_src1_nunique', 'amt_src1_count',
       'amt_src2_nunique', 'amt_src2_count', 'device1_nunique',
       'device1_count', 'device2_nunique', 'device2_count', 'mac1_nunique',
       'mac1_count', 'geo_code_nunique', 'geo_code_count', 'ip1_sub_nunique',
       'ip1_sub_count', 'ip1_nunique', 'ip1_count', 'acc_id1_nunique',
       'acc_id1_count', 'device_code1_nunique', 'device_code1_count',
       'trans_type1_nunique', 'trans_type1_count', 'trans_type2_nunique',
       'trans_type2_count']
for col in cols:
    print(col)
    op_test3 = get_category_feature(test_x2, col, op_test3)
print(op_test3.shape)


# In[ ]:


aa = op_train3
aaa = aa.iloc[:,2:]
qq = np.array(aaa)
pp = pd.DataFrame(qq)
aaaa = aa.iloc[:,0:2]
aaaaa = pd.concat([aaaa,pp],axis=1)
cc = 0
for col in aaaaa.columns[2:]:
    aaaaa.rename(columns= {col: 'feature' + '_'+ str(cc)}, inplace=True)
    cc = cc + 1

bb = op_test3
bbb = bb.iloc[:,2:]
qqq = np.array(bbb)
ppp = pd.DataFrame(qqq)
bbbb = bb.iloc[:,0:2]
bbbbb = pd.concat([bbbb,ppp],axis=1)
ccc = 0
for col in bbbbb.columns[2:]:
    bbbbb.rename(columns= {col: 'feature' + '_'+ str(ccc)}, inplace=True)
    ccc = ccc + 1  

op_train4 = aaaaa
op_test4 = bbbbb

op_test4.Tag = -1

all_data2 = pd.concat([op_train4, op_test4])


# In[ ]:


all_data2.head()


# In[ ]:


train_data = all_data2[all_data2['Tag']!=-1]
test_data = all_data2[all_data2['Tag']==-1]
print(train_data.shape)
print(test_data.shape)


# In[ ]:


train_data = train_data.fillna(-1)
test_data = test_data.fillna(-1)


# In[ ]:


test_data.head()


# In[ ]:


nu_train = train_data.isnull().any()
for i in range(509):
    if nu_train[i] == True:
        print(i)
print('*************************************')
nu_test = test_data.isnull().any()
for i in range(509):
    if nu_test[i] == True:
        print(i)


# In[ ]:


X_tr = train_data.drop(['Tag'],axis = 1)
y_tr = train_data['Tag']

test_id = test_data['UID']
X_test = test_data.drop(['Tag'],axis = 1)

print(X_tr.shape)
print(y_tr.shape)
print(X_test.shape)


# # predict

# In[ ]:


# def get_top_n_features(train_data_X, train_data_Y, top_n_features):

#     # random forest
#     rf_est = RandomForestClassifier(random_state=0)
#     rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
#     rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=10, cv=5, verbose=1)
#     rf_grid.fit(train_data_X, train_data_Y)
#     print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
#     print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
#     print('Top N Features RF Train Score:' + str(rf_grid.score(train_data_X, train_data_Y)))
#     feature_imp_sorted_rf = pd.DataFrame({'feature': list(train_data_X),
#                                           'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
#     features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
#     print('Sample 10 Features from RF Classifier')
#     print(str(features_top_n_rf[:10]))

#     # AdaBoost
#     ada_est =AdaBoostClassifier(random_state=0)
#     ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
#     ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=10, cv=5, verbose=1)
#     ada_grid.fit(train_data_X, train_data_Y)
#     print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
#     print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
#     print('Top N Features Ada Train Score:' + str(ada_grid.score(train_data_X, train_data_Y)))
#     feature_imp_sorted_ada = pd.DataFrame({'feature': list(train_data_X),
#                                            'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
#     features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
#     print('Sample 10 Feature from Ada Classifier:')
#     print(str(features_top_n_ada[:10]))

#     # DecisionTree
#     dt_est = DecisionTreeClassifier(random_state=0)
#     dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
#     dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=10, cv=5, verbose=1)
#     dt_grid.fit(train_data_X, train_data_Y)
#     print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
#     print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
#     print('Top N Features DT Train Score:' + str(dt_grid.score(train_data_X, train_data_Y)))
#     feature_imp_sorted_dt = pd.DataFrame({'feature': list(train_data_X),
#                                           'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
#     features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
#     print('Sample 10 Features from DT Classifier:')
#     print(str(features_top_n_dt[:10]))

#     # merge the three models
#     features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_dt], 
#                                ignore_index=True).drop_duplicates()

#     features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada,feature_imp_sorted_dt],ignore_index=True)

#     return features_top_n , features_importance


# In[ ]:


# feature_to_pick = 400
# feature_top_n, feature_importance = get_top_n_features(X_tr, y_tr, feature_to_pick)


# In[ ]:


# feature_top_n =  feature_importance[feature_importance.importance>0.00125].feature.unique()


# In[ ]:


# feature_importance.to_csv('../data/feature1/feature_importance.csv',index=False)


# In[ ]:


# train_data_X = pd.DataFrame(X_tr[feature_top_n])
# test_data_X = pd.DataFrame(X_test[feature_top_n])


# In[ ]:


# train_data_X.shape


# In[ ]:


def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3


# In[ ]:


# X_train,X_val,y_train,y_val= train_test_split(train_data_X,y_tr,test_size=0.2,random_state=2)


# In[ ]:


X_train,X_val,y_train,y_val= train_test_split(X_tr,y_tr,test_size=0.2,random_state=8)


# In[ ]:


lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)


# In[ ]:


param = {
        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': 'auc',

        'min_child_weight': 1.5,

        'num_leaves': 2 ** 5,

        'lambda_l2': 10,

        'subsample': 0.85,

        'learning_rate': 0.01,

        'seed': 88,

        'colsample_bytree': 0.5,

        'nthread': 12
}

param['is_unbalance']='true'

bst=lgb.cv(param,lgb_train, num_boost_round=500, nfold=5, early_stopping_rounds=50)
gbm = lgb.train(param,lgb_train,num_boost_round=len(bst['auc-mean']))


# In[ ]:


ypred = gbm.predict(X_val)
score = tpr_weight_funtion(y_predict=ypred,y_true=y_val)
print('score:',score)


# In[ ]:


# y_pred = gbm.predict(test_data_X)


# In[ ]:


y_pred = gbm.predict(X_test)


# In[ ]:


sub = pd.read_csv('../../01-data/提交样例.csv')
sub['Tag'] = y_pred
sub.to_csv('./model_1.csv',index=False)

