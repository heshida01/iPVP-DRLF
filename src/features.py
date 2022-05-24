#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time : 2022.05.18
# @Author : jiaoshihu
# @Email : shihujiao@163.com
# @IDE : PyCharm
# @File : iPVP_DRLP.py


from src.UniRep_embedding import UniRep_Embed
from src.BiLSTM_embedding import BiLSTM_Embed
from src.DPC import get_DPC
from src.DDE import get_DDE
import pandas as pd
import numpy as np

def select_features(allfeatures,feature_index):
    new_features = []
    orignal_data = pd.DataFrame(allfeatures)
    for i in list(feature_index):
        new_features.append(orignal_data[int(i)])
    features = np.array(new_features).T
    return features

def get_feature(fastas):
    encoding = []
    feature_index = pd.read_csv(r"feature_index.csv", header=None)

    DDE_features = get_DDE(fastas)
    new_DDE_feature = select_features(DDE_features, feature_index.iloc[0, :38])
    encoding.append(new_DDE_feature)

    BiLSTM_features = BiLSTM_Embed(fastas)
    new_BiLSTM_features = select_features(BiLSTM_features, feature_index.iloc[1, :44].values)
    encoding.append(new_BiLSTM_features)

    DPC_features = get_DPC(fastas)
    new_DPC_feature = select_features(DPC_features, feature_index.iloc[2, :34].values)
    encoding.append(new_DPC_feature)

    UniRep_features = UniRep_Embed(fastas)
    new_UniRep_features = select_features(UniRep_features, feature_index.iloc[3, :60].values)
    encoding.append(new_UniRep_features)

    encoding = np.column_stack(encoding)
    new_encoding = select_features(encoding, feature_index.iloc[4, :63].values)

    return new_encoding

