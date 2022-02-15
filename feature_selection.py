#!/usr/bin/env python
#_*_coding:utf-8_*_


import numpy as np
import pandas as pd

ASDC_index = pd.read_csv(r"ASDC_index.csv")
UniRep_index = pd.read_csv(r"UniRep_index.csv")
def select_ASDC_features(allfeatures):
    print('Feature selection...')
    new_features = []
    orignal_data = pd.DataFrame(allfeatures)
    for i in list(ASDC_index):
        new_features.append(orignal_data[int(i)])
    features = np.array(new_features).T

    return features



def select_UniRep_features(allfeatures):
    print('Feature selection...')
    new_features = []
    orignal_data = pd.DataFrame(allfeatures)
    for i in list(UniRep_index):
        new_features.append(orignal_data[int(i)])
    features = np.array(new_features).T

    return features



