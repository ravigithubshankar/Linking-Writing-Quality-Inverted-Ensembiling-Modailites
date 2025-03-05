from baseline import Yunbase
import polars as pl#similar to pandas, but with better performance when dealing with large datasets.
import pandas as pd#read csv,parquet
import numpy as np#for scientific computation of matrices
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer#word2vec feature
import warnings#avoid some negligible errors
#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')

import random#provide some function to generate random_seed.
#set random seed,to make sure model can be recurrented.
def seed_everything(seed):
    np.random.seed(seed)#numpy's random seed
    random.seed(seed)#python built-in random seed
seed_everything(seed=2025)

from tqdm import tqdm
print("< load dataset >")
train_feats=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv")
test_feats=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/sample_submission.csv")

train_df=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv")
test_df=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv")

print("< train features >")
feats=[]
for id in tqdm(train_feats['id'].values):
    tmp_df=train_df[train_df['id']==id].drop(['id'],axis=1)
    feat=[]
    for col in tmp_df.columns:
        if tmp_df[col].dtype==object:
            feat.append(" ".join(tmp_df[col].values))
        else:#float
            feat.append(tmp_df[col].values)
    feats.append(feat)
feats=pd.DataFrame(feats)
feats.columns=train_df.drop(['id'],axis=1).columns
train_feats=pd.concat((train_feats,feats),axis=1)

print("< test features >")
feats=[]
for id in tqdm(test_feats['id'].values):
    tmp_df=test_df[test_df['id']==id].drop(['id'],axis=1)
    feat=[]
    for col in tmp_df.columns:
        if tmp_df[col].dtype==object:
            feat.append(" ".join(tmp_df[col].values))
        else:#float
            feat.append(tmp_df[col].values)
    feats.append(feat)
feats=pd.DataFrame(feats)
feats.columns=test_df.drop(['id'],axis=1).columns
test_feats=pd.concat((test_feats,feats),axis=1)


def FE(df):
    df['event_id_max']=df['event_id'].apply(lambda x:np.max(x))
    df.drop(['event_id'],axis=1,inplace=True)
    return df
train_feats=FE(train_feats)
test_feats=FE(test_feats)

train_feats.head()


