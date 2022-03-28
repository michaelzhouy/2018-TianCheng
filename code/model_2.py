#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[2]:


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


# In[3]:


path = '../01-data'

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


# In[4]:


def get_operation_features(tag_data, operation_data):
    columns = ['day', 'mode', 'success', 'time', 'os', 'version', 'device1',
               'device2', 'device_code1', 'device_code2', 'device_code3', 'mac1',
               'mac2', 'ip1', 'wifi', 'geo_code', 'ip1_sub']
    delete = ['mode', 'time', 'os', 'device1', 'success']
    
    operation_data['time'] = operation_data['time'].apply(lambda x: int(x.split(':')[0]))
    tmp_time = operation_data[operation_data['time']<6]
    tmp_time = tmp_time.groupby(['UID'], as_index=False)['time'].agg({
        'op_time_abnor_nunique':'nunique',
        'op_time_abnor_count':'count'
    })
    tag_data = pd.merge(tag_data, tmp_time, on='UID', how='left')

    operation_data['isnull_num'] = operation_data.apply(lambda x: x.isnull().sum(),axis=1)
    agg_dict = {'isnull_num_max':'max', 'isnull_num_min':'min', 'isnull_num_mean':'mean', 'isnull_numt_std':'std', 'isnull_numt_sum':'sum'}
    tag_data = tag_data.merge(operation_data.groupby(['UID'], as_index=False)['isnull_num'].agg(agg_dict), on='UID', how='left')

    for col in columns:
        count_column = col+'_op_count'
        nunique_column = col+'_op_nunique'
        if col in delete:
            agg_dict = {nunique_column: 'nunique'}
        else:
            agg_dict = {count_column: "count", nunique_column: 'nunique'}
        
        tag_data = tag_data.merge(operation_data.groupby(['UID'], as_index=False)[col].agg(agg_dict), on='UID', how='left')

        count_column = col+'_UID_op_count'
        agg_dict = {count_column: "count"}
        tmp_cat =  operation_data.groupby(['UID', col], as_index=False)['Tag'].agg(agg_dict)
        agg_dict = {col+'_op_count_mean':'mean', col+'_op_count_std':'std'}
        tag_data = tag_data.merge(tmp_cat.groupby(['UID'], as_index=False)[count_column].agg(agg_dict), on='UID', how='left')

    one_columns = []
    for col in columns:
        isnull_  = operation_data[col].isnull().sum()/len(operation_data)
        nunique_ = operation_data[col].nunique()
        if nunique_ > 200:
            one_columns.append(col)

    for col in one_columns:
        nunique_ = operation_data[col].nunique()
        if nunique_<5000:
            threshold = 10
        elif nunique_<10000 and nunique_>=5000:
            threshold = 8
        elif nunique_<16000 and nunique_>=10000:
            threshold = 5
        else:
            threshold = 3
        tmp_all = operation_data.groupby(['UID', col], as_index=False)['Tag'].agg({col+'_count':'count'})
        tmp = tmp_all.groupby([col], as_index=False)['UID'].agg({col+'_UID_count':'count'})
        
        #tmp_return = pd.merge(tmp_all, tmp, on=col, how=left)
        #agg_dict = {col+'_num_max':'max', col+'_num_min':'min', col+'_num_mean':'mean', col+'_numt_std':'std', col+'_numt_sum':'sum'}
        #tag_data = tag_data.merge(tmp_return.groupby(['UID'])[col+'_UID_count'].agg(agg_dict).reset_index(), on='UID', how='left')
        
        tmp = tmp[tmp[col+'_UID_count']>threshold+3]
        tmp[col+'_UID_count'] = 1
        tmp_all = pd.merge(tmp_all, tmp, on=col, how='left')
        tmp_all = tmp_all.fillna(0)
        tmp_all = tmp_all.groupby(['UID'], as_index=False)[col+'_UID_count'].agg({col+'_UID_sum':'sum'})
        tag_data = pd.merge(tag_data, tmp_all, on='UID', how='left')


    return tag_data


# In[5]:


def get_transaction_features(tag_data, transaction_data):
    num_columns = ['trans_amt', 'bal']
    columns = ['channel', 'day', 'time', 'amt_src1', 'merchant',
                   'trans_type1', 'acc_id1', 'device_code1',
                   'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
                   'amt_src2', 'acc_id2', 'acc_id3', 'geo_code', 'trans_type2',
                   'market_code', 'market_type', 'ip1_sub']
    delete = ['day', 'time', 'amt_src1', 'merchant','trans_type1', 'ip1', 'bal']

    transaction_data['time'] = transaction_data['time'].apply(lambda x: int(x.split(':')[0]))
    tmp_time = transaction_data[transaction_data['time']<6]
    tmp_time = tmp_time.groupby(['UID'], as_index=False)['time'].agg({'trans_time_abnor_nunique':'nunique', 'trans_time_abnor_count':'count'})
    tag_data = pd.merge(tag_data, tmp_time, on='UID', how='left')

    transaction_data['isnull_num'] = transaction_data.apply(lambda x: x.isnull().sum(),axis=1)
    agg_dict = {'isnull_num_max':'max', 'isnull_num_min':'min', 'isnull_num_mean':'mean', 'isnull_numt_std':'std', 'isnull_numt_sum':'sum'}
    tag_data = tag_data.merge(transaction_data.groupby(['UID'], as_index=False)['isnull_num'].agg(agg_dict), on='UID', how='left')

    for col in columns:
        count_column = col+'_trans_count'
        nunique_column = col+'_trans_nunique'
        if col in delete:
            agg_dict = {nunique_column: 'nunique'}
        else:
            agg_dict = {count_column: "count", nunique_column: 'nunique'}
        
        tag_data = tag_data.merge(transaction_data.groupby(['UID'], as_index=False)[col].agg(agg_dict), on='UID', how='left')

        count_column = col+'_UID_trans_count'
        agg_dict = {count_column: "count"}
        tmp_cat =  transaction_data.groupby(['UID', col], as_index=False)['Tag'].agg(agg_dict)
        agg_dict = {col+'_trans_count_mean':'mean', col+'_trans_count_std':'std'}
        tag_data = tag_data.merge(tmp_cat.groupby(['UID'], as_index=False)[count_column].agg(agg_dict), on='UID', how='left')

    for col in num_columns:
        nunique_column = col+'_trans_nunique'
        max_column = col+'_trans_max'
        min_column = col+'_trans_min'
        sum_column = col+'_trans_sum'
        mean_column = col+'_trans_mean'
        std_column = col+'_trans_std'

        agg_dict = {nunique_column:'nunique', max_column:'max', min_column:'min', sum_column:'sum', mean_column:'mean', std_column:'std'}

        tag_data = tag_data.merge(transaction_data.groupby(['UID'], as_index=False)[col].agg(agg_dict), on='UID', how='left')

    one_columns = []
    for col in columns:
        isnull_  = transaction_data[col].isnull().sum()/len(transaction_data)
        nunique_ = transaction_data[col].nunique()
        if nunique_ > 200:
            one_columns.append(col)
    delete = delete+one_columns
    one_columns = columns
    columns = [i for i in columns if i not in delete]

    ##10 8 5 3 0.653
    for col in one_columns:
        nunique_ = transaction_data[col].nunique()
        if nunique_<5000:
            threshold = 10
        elif nunique_<10000 and nunique_>=5000:
            threshold = 8
        elif nunique_<16000 and nunique_>=10000:
            threshold = 5
        else:
            threshold = 3
        tmp_all = transaction_data.groupby(['UID',col], as_index=False)['Tag'].agg({col+'_count':'count'})
        tmp = tmp_all.groupby([col], as_index=False)['UID'].agg({col+'_UID_count':'count'})
        tmp = tmp[tmp[col+'_UID_count']>threshold+3]
        tmp[col+'_UID_count'] = 1
        tmp_all = pd.merge(tmp_all, tmp, on=col, how='left')
        tmp_all = tmp_all.fillna(0)
        tmp_all = tmp_all.groupby(['UID'], as_index=False)[col+'_UID_count'].agg({col+'_UID_sum':'sum'})
        tag_data = pd.merge(tag_data, tmp_all, on='UID', how='left')

    return tag_data


# In[6]:


def get_ctr_feature(tag_data):
    train_data = tag_data[tag_data['Tag']!=-1]
    columns = tag_data.columns
    items = []
    lack_features = ['device_code3','mac1','acc_id2','acc_id3','market_code','market_type']
    for col in columns:
        if tag_data[col].nunique()<30 and col != 'Tag':
            flag=0
            for lack_ in lack_features:
                if lack_ in col:
                    flag=1
                    break
            if flag==0:
                items.append(col)
    for item in items:
        temp = train_data.groupby(item, as_index = False)['Tag'].agg({item+'_click':'sum', item+'_count':'count'})
        temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count'])
        tag_data = pd.merge(tag_data, temp, on=item, how='left')

    for i in range(len(items)):
        for j in range(i+1, len(items)):
            item_g = [items[i], items[j]]
            temp = train_data.groupby(item_g, as_index=False)['Tag'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
            temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
            tag_data = pd.merge(tag_data, temp, on=item_g, how='left')

    return tag_data


# In[7]:


def get_cat_used(tag_data, operation_data, transaction_data):
    
    def get_cat(x):
        res = []
        tmp = x.value_counts()
        tmp_value=list(tmp)
        if tmp.shape[0]>1 and tmp.index[0]==-1:
            res.append(tmp.index[1])
            res.append(tmp_value[1]/len(x))
            return res
        else:
            res.append(tmp.index[0])
            res.append(tmp_value[0]/len(x))
            return res


    cols = ['mode', 'os',  'device2',  'geo_code', 'ip1_sub','version',  'mac2','device1', 'device_code1', 'device_code2', 'device_code3','ip1', 'mac1','wifi']
    for col in cols:
        tmp=pd.DataFrame()
        operation_data[col] = pd.factorize(operation_data[col])[0]
        tmp = operation_data.groupby(['UID'])[col].apply(get_cat).reset_index()
        #tmp[col+'_op_clike'] = tmp[col].apply(lambda x: x[0])
        tmp[col+'_op_clike_ratio'] = tmp[col].apply(lambda x: x[1])
        tmp = tmp.drop(col, axis=1)
        tag_data = pd.merge(tag_data, tmp, on='UID',how='left')

    #for col in cols:
        #tag_data = tag_data.merge(operation_data.groupby(['UID'])[col].agg(lambda x: x.value_counts().index[0]).reset_index(),on='UID',how='left')

    #op_data['hour'] = op_data.time.str[0:2].astype(int)
    #op_data['time'] = pd.to_datetime(op_data['time'])
    #op_data['time'] = op_data['time'].apply(lambda x:(x-datetime.now()).seconds/3600)

    cols_f = ['amt_src1', 'trans_type1','amt_src2','geo_code','ip1_sub','mac1','acc_id2','merchant','acc_id3','channel','market_type',
            'trans_type2','market_code','ip1','device_code2','device_code3','device1','device_code1','acc_id1','device2']
    for col in cols_f:
        tmp=pd.DataFrame()
        transaction_data[col] = pd.factorize(transaction_data[col])[0]
        tmp = transaction_data.groupby(['UID'])[col].apply(get_cat).reset_index()
        #tmp[col+'_trans_clike'] = tmp[col].apply(lambda x: x[0])
        tmp[col+'_trans_clike_ratio'] = tmp[col].apply(lambda x: x[1])
        tmp = tmp.drop(col, axis=1)
        tag_data = pd.merge(tag_data, tmp, on='UID',how='left')

    #for col in cols_f:
        #tag_data = tag_data.merge(transaction_data.groupby(['UID'])[col].agg(lambda x: x.value_counts().index[0]).reset_index(),on='UID',how='left')
    
    #tr_data['hour'] = tr_data.time.str[0:2].astype(int)
    #tr_data['time'] = pd.to_datetime(tr_data['time'])
    #tr_data['time'] = tr_data['time'].apply(lambda x:(x-datetime.now()).seconds/3600)

    return tag_data


# In[8]:


def bulid_cat_feature(uid_list,feature_list,featureName):
    #1.找到每个uid有哪些特征
    uid2featureList = {}  #key uid  feature 这个uid对应历史上出现过哪些list
    for i in range(len(uid_list)):
        uid = uid_list[i]
        feature = feature_list[i]
        #不为空
        #if feature:
        if not pd.isnull(feature):
            if uid not in uid2featureList:
                uid2featureList[uid] = set()
            uid2featureList[uid].add(str(feature))  #用str

    feasvalue2index = {}  #这个值对应的特征名         
    #2.将每一行转化为dict
    rows = []
    for uid,value in uid2featureList.items():
        tmp_dict = {}
        tmp_dict["UID"] = uid
        value = "@".join(sorted(list(value)))
        if value not in feasvalue2index:
            feasvalue2index[value] = len(feasvalue2index)
        tmp_dict[featureName+"_cat"] = feasvalue2index[value]
        rows.append(tmp_dict)
    df = pd.DataFrame(rows)
    return df


# In[9]:


def get_cat_feature(tag_data, operation_data, transaction_data):
    one_hot_t_feature = ["channel","amt_src1","trans_type1","amt_src2","trans_type2","market_type"]
    one_hot_op_feature = ["mode","success","os","version"]
    #4.构造分类特征 0.31 -> 0.48
    for feature in one_hot_t_feature:
        df_temp = bulid_cat_feature(list(transaction_data["UID"]),list(transaction_data[feature]),feature)
        tag_data = tag_data.merge(df_temp,on='UID',how='left')

    for feature in one_hot_op_feature:
        df_temp = bulid_cat_feature(list(operation_data["UID"]),list(operation_data[feature]),feature)
        tag_data = tag_data.merge(df_temp,on='UID',how='left')

    return tag_data


# In[10]:


def get_cross_feature(tag_data, operation_data, transaction_data):
    cross_col = ['mode', 'os', 'version', 'device1',
                'device2', 'device_code1', 'device_code2', 'device_code3', 'mac1',
                'mac2', 'ip1',  'wifi', 'geo_code', 'ip1_sub']
    for col in cross_col: 
        tmp = operation_data.groupby(['UID', 'day'], as_index=False)[col].agg({col+'_op_day_nunique': 'nunique'})
        tmp = tmp.groupby(['UID'], as_index=False)[col+'_op_day_nunique'].agg({col+'_op_day_nunique_max':'max', col+'_op_day_nunique_mean':'mean'})
        tag_data = pd.merge(tag_data, tmp, on='UID', how='left')
    cross_col = ['channel', 'trans_amt', 'amt_src1', 'merchant',
                'trans_type1', 'acc_id1', 'device_code1',
                'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
                'bal', 'amt_src2', 'acc_id2', 'acc_id3', 'geo_code', 'trans_type2',
                'market_code', 'market_type', 'ip1_sub']
    for col in cross_col: 
        tmp = transaction_data.groupby(['UID', 'day'], as_index=False)[col].agg({col+'_trans_day_nunique': 'nunique'})
        tmp = tmp.groupby(['UID'], as_index=False)[col+'_trans_day_nunique'].agg({col+'_trans_day_nunique_max':'max', col+'_trans_day_nunique_mean':'mean'})
        tag_data = pd.merge(tag_data, tmp, on='UID', how='left')
    return tag_data


# In[11]:


def get_time_feature(tag_data, operation_data, transaction_data):
    """
feature_prefix:特征前缀

对day特征进行构建
针对每个UID
(1)计算这个UID前后两次交易的平均时间   如  1,6,10  那么平均时间为  （5 + 4） / 2  = 4.5 (多少天会交易1次)
(2)计算这个UID第1次交易和最后一次交易的时间差   如1,6,10  那么前后两次交易的时间差为 10 - 1 = 9

后续可以开发的 max_gap  min_gap 

"""
    def bulid_day_avg_gap(uid_list,feature_list,feature_prefix):
        uid2daylist = {}
        #1.先计算每个uid有哪些day
        for i in range(len(uid_list)):
            uid = uid_list[i]
            day = feature_list[i]
            if uid not in uid2daylist:
                uid2daylist[uid] = []
            uid2daylist[uid].append(day)

        rows = []
        #2.对每个uid进行处理
        for uid,value in uid2daylist.items():
            value = sorted(value)
            if len(value) == 1:
                avg = 0
                gap = 0
            else:
                tmp_sum = 0
                for i in range(1,len(value)):
                    tmp_sum += (value[i] - value[i - 1])
                avg = tmp_sum * 1.0 / (len(value))
                gap = value[-1] - value[0]

            tmp_dict = {}
            tmp_dict["UID"] = uid
            tmp_dict[feature_prefix + "_avg"] = avg
            tmp_dict[feature_prefix + "_gap"] = gap
            rows.append(tmp_dict)

        df = pd.DataFrame(rows)
        #if norm_flag:
        #归一化
        #df[feature_prefix + "_avg"] = norm_list(list(df[feature_prefix + "_avg"]))
        #df[feature_prefix + "_gap"] = norm_list(list(df[feature_prefix + "_gap"]))
        #print df
        return df

    """
    对time进行处理
    (1)计算这个UID前后两次交易的平均时间   
    (2)计算这个UID第1次交易和最后一次交易的时间差
    _day_hour 精确到小时
    _day_hour_min 精确到分钟
    """
    def bulid_time_avg_gap(uid_list,day_list,time_list,feature_prefix):
        day_hour_list = []
        day_hour_min_list = []
        for i in range(len(day_list)):
            day = int(day_list[i])
            time_array_2 = time_list[i].split(':') #时分秒
            hour = int(time_array_2[0])
            minute = int(time_array_2[1])
            day_hour_list.append(day * 24 + hour )
            day_hour_min_list.append(day * 3600 + hour * 60 + minute)

        df1 = bulid_day_avg_gap(uid_list,day_hour_list,feature_prefix+"_day_hour")
        df2 = bulid_day_avg_gap(uid_list,day_hour_min_list,feature_prefix+"_day_hour_min")
        df1 = df1.merge(df2,on='UID',how='left')
        #print df1
        return df1
    #2.时间day的统计 (这部分如果不需要基础统计，就不放在get_feature之前)
    #统计每个UID 每次交易的平均时间间隔 最大时间间隔，最小时间间隔 (Day为单位)
    #(1)trans - day
    df_temp = bulid_day_avg_gap(list(transaction_data["UID"]),list(transaction_data["day"]),"trans_day_between")
    tag_data = tag_data.merge(df_temp,on='UID',how='left')

    #(2)op - day
    df_temp = bulid_day_avg_gap(list(operation_data["UID"]),list(operation_data["day"]),"op_day_between")
    tag_data = tag_data.merge(df_temp,on='UID',how='left')

    #(3)trans - time
    #df_temp = bulid_time_avg_gap(list(transaction_data["UID"]),list(transaction_data["day"]),list(transaction_data["time"]),"trans_time_between")
    #tag_data = tag_data.merge(df_temp,on='UID',how='left')

    #(4)op - time
    #df_temp = bulid_time_avg_gap(list(operation_data["UID"]),list(operation_data["day"]),list(operation_data["time"]),"op_time_between")
    #tag_data = tag_data.merge(df_temp,on='UID',how='left')

    return tag_data


# In[12]:


def deal_tag_features(tag_data):
    features = tag_data.columns
    flag = '_Tag_count_max'
    Tag_features = [col for col in features if flag in col]
    tag_data = tag_data.drop(Tag_features, axis=1)
    flag = 'ctr'
    Tag_features = [col for col in features if flag in col]
    tag_data = tag_data.drop(Tag_features, axis=1)
    #tag_data.loc[tag_data['Tag_count_max_sum']>3, 'Tag_count_max_sum']=4
    return tag_data


# In[13]:


def get_day_feature(data_set, tag_data):
    tmp = data_set.groupby(['UID', 'merchant'], as_index=False)['day'].agg({'day':'mean'})
    tmp_merchant_day = tmp.groupby(['merchant', 'day'], as_index=False)['UID'].agg({'count':'count'})
    tmp_merchant = tmp.groupby(['merchant'], as_index=False)['UID'].agg({'all_count':'count'})
    tmp_merchant_day = tmp_merchant_day.merge(tmp_merchant, on='merchant', how='left')
    tmp_merchant_day['day_ratio'] = tmp_merchant_day['count']/tmp_merchant_day['all_count']
    tmp = tmp.merge(tmp_merchant_day, on=['merchant', 'day'], how='left')
    tmp = tmp.groupby(['UID'], as_index=False)['day_ratio'].agg({'ratio_mean':'mean', 'ratio_max':'max', 'ratio_min':'min', 'ratio_std':'std'})
    tag_data = tag_data.merge(tmp, on='UID', how='left')
    return tag_data

def get_mercant_feature(data_set, tag_data):
    tmp = data_set.groupby(['merchant'], as_index=False)['UID'].agg({'merchant_UID_nunique':'nunique'})
    tmp = data_set.merge(tmp, on='merchant', how='left')
    tmp = tmp.groupby(['UID'], as_index=False)['merchant_UID_nunique'].agg({
        'merchant_UID_nunique_max': 'max',
        'merchant_UID_nunique_mim': 'min',
        'merchant_UID_nunique_mean': 'mean',
        'merchant_UID_nunique_std': 'std'
    })
    tag_data = tag_data.merge(tmp, on='UID', how='left')
    return tag_data


# In[14]:


def get_features(tag_data, operation_data, transaction_data):
    print('==============================')
    print('start getting operation features....')
#
    tag_data = get_operation_features(tag_data, operation_data)
#
    print('finish getting operation features....')
#
    print('==============================')
    print('start getting transaction features....')
#
    tag_data = get_transaction_features(tag_data, transaction_data)
    
#
    print('finish getting transaction features....')
#
    print('==============================')
    print('start getting ctr features....')
#
    #tag_data = get_ctr_feature(tag_data)
#    
    #tag_data = get_checkBlack_count(tag_data, operation_data, transaction_data)

    print('finish getting ctr features....')

    print('==============================')
    print('start getting cat features....')

    #tag_data = get_cat_feature(tag_data, operation_data, transaction_data)

    tag_data = get_time_feature(tag_data, operation_data, transaction_data)

    tag_data = get_cross_feature(tag_data, operation_data, transaction_data)

    tag_data = get_cat_used(tag_data, operation_data, transaction_data)

    tag_data = get_day_feature(transaction_data, tag_data)

    #tag_data = get_mercant_feature(transaction_data, tag_data)
    print('start getting cat features....')
    #tag_data = pd.read_csv('../data/tag_data.csv')

    tag_data = deal_tag_features(tag_data)

    return tag_data


# In[15]:


def lgb_model(tag_data, operation_data, transaction_data):


    #train_features = train_data[features_columns]
    #train_labels = train_data["tag"]

    lgb_parms = {
                "boosting_type": "gbdt",
                "num_leaves": 35,
                "max_depth": -1,
                "learning_rate": 0.1,
                "n_estimators": 500,
                "max_bin": 425,
                "subsample_for_bin": 20000,
                "objective": 'binary',
                "metric": 'auc',
                "min_split_gain": 0,
                "min_child_weight": 0.001,
                "min_child_samples": 20,
                "subsample": 0.9,
                "subsample_freq": 1,
                "colsample_bytree": 0.7,
                "reg_alpha": 3,
                "reg_lambda": 5,
                "seed": 2018,
                "n_jobs": 25,
                "verbose": 1,
                "silent": False,
                }

    #test_data = test_data.drop(['Tag'], axis = 1)

    trian_tag = tag_data[tag_data['Tag']!=-1]
    test_tag = tag_data[tag_data['Tag']==-1]

    n_folds = 2

    preds_list = list()
    vali_score = list()
    features_importance=pd.DataFrame()
    for times in range(n_folds):
        train_tag, val_tag, train_tag_, val_tag_ = train_test_split(trian_tag,trian_tag['Tag'],test_size=0.5,random_state=times)
        test_tag_1, test_tag_2, test_1, test_tag2 = train_test_split(test_tag,test_tag['Tag'],test_size=0.5,random_state=times)
        #train_data
        train_op_data = operation_data[operation_data['UID'].isin(train_tag['UID'])]
        train_trans_data = transaction_data[transaction_data['UID'].isin(train_tag['UID'])]
        #val_data
        val_op_data = operation_data[operation_data['UID'].isin(val_tag['UID'])]
        val_trans_data = transaction_data[transaction_data['UID'].isin(val_tag['UID'])]
        #test_data1
        test_op_data_1 = operation_data[operation_data['UID'].isin(test_tag_1['UID'])]
        test_trans_data_1 = transaction_data[transaction_data['UID'].isin(test_tag_1['UID'])]
        #test_data2
        test_op_data_2 = operation_data[operation_data['UID'].isin(test_tag_2['UID'])]
        test_trans_data_2 = transaction_data[transaction_data['UID'].isin(test_tag_2['UID'])]

        train_tag = get_features(train_tag, train_op_data, train_trans_data)
        val_tag = get_features(val_tag, val_op_data, val_trans_data)
        test_tag_1 = get_features(test_tag_1, test_op_data_1, test_trans_data_1)
        test_tag_2 = get_features(test_tag_2, test_op_data_2, test_trans_data_2)

        test_data = pd.concat([test_tag_1, test_tag_2], axis=0, ignore_index=True, sort=False)
        test_data = test_data.sort_values(['UID'])

        print(test_data['UID'])
        print(train_tag.shape)
        print(val_tag.shape)
        print(test_data.shape)

        print('==============================')
        print('start training....')

        columns = train_tag.columns
        remove_columns = ['Tag']
        features_columns = [column for column in columns if column not in remove_columns]
        
        test_data = test_data.drop(['Tag'], axis = 1)
        features_importance['feature_name'] = features_columns
        for i in range(2):
            if i==0:
                k_train = train_tag
                k_test = val_tag
            else:
                k_train = val_tag
                k_test = train_tag


            X_train = k_train.drop(['Tag'], axis = 1)
            y_train = k_train['Tag']
            X_test = k_test.drop(['Tag'], axis = 1)
            y_test = k_test['Tag']

            gbm = lgb.LGBMClassifier(**lgb_parms)
            gbm = gbm.fit(X_train, y_train,
                          eval_metric="auc",
                          eval_set=[(X_train, y_train),
                                    (X_test, y_test)],
                          eval_names=["train", "valid"],
                          early_stopping_rounds=100,
                          verbose=True)

            vali_pred = gbm.predict_proba(X_test, num_iteration=gbm.best_iteration_)[:, 1]
            vali_k_score = tpr_weight_funtion(y_test, vali_pred)

            print('The {} {} kfold score : {}'.format(times, i, vali_k_score))

            vali_score.append(vali_k_score)
            preds = gbm.predict_proba(test_data, num_iteration=gbm.best_iteration_)[:, 1]

            preds_list.append(preds)

            booster = gbm.booster_
            importance = booster.feature_importance(importance_type='split')
            feature_name = booster.feature_name()
            feature_importance = pd.DataFrame({'feature_name':feature_name,'importance_'+str(times)+str(i):importance} )
            features_importance = pd.merge(features_importance, feature_importance, how='left', on='feature_name')

    features_importance.to_csv('./features_importance_devide.csv', index=False)

    for n, (i) in enumerate(vali_score):
        print('score_{}: {}'.format(n, i))    
    print('score_mean: ', np.mean(vali_score))

    s = 0
    for i in preds_list:
        s = s + i

    test_tag = test_tag.drop(['Tag'], axis = 1)
    print(s)
#     save = pd.DataFrame(list(s))
#     save.to_csv('../submit/save.csv',index = False)
    test_tag['Tag'] = list(s / (n_folds*2))
    test_tag[['UID', 'Tag']].to_csv('./model_2.csv', index=False)


# In[16]:


def Processing():
    operation_data = get_data(name='operation')
    transaction_data = get_data(name='transaction')
    tag_data = get_data(name='tag')
    operation_data = pd.merge(operation_data, tag_data, on='UID', how='left')
    transaction_data = pd.merge(transaction_data, tag_data, on='UID', how='left')
    op_delete=['ip2', 'ip2_sub' ]
    trans_delete = ['code1', 'code2']
    operation_data = operation_data.drop(op_delete, axis=1)
    transaction_data = transaction_data.drop(trans_delete, axis=1)
    print('finish get data...')

    lgb_model(tag_data, operation_data, transaction_data)


# In[ ]:


if __name__ == "__main__":
    Processing()

