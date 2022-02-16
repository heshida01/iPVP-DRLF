#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time : 2022.12.02
# @Author : jiaoshihu
# @Email : shihujiao@163.com
# @IDE : PyCharm
# @File : iPVP_DRLP.py

from __future__ import print_function,division

import read_fasta_sequences
import numpy as np
import pandas as pd
import joblib
import feature_selection
import argparse
import time
import os
import collections
import warnings
from feature_scripts.UniRep_emb import UniRep_Embed
from feature_scripts.ASDC import get_ASDC

warnings.filterwarnings('ignore')

if not os.path.exists("results"):
    os.makedirs("results")

def predict(fastas):
    encoding = []
    print("Sequence encoding......")
    UniRep_features = UniRep_Embed(fastas)
    new_UniRep_features = feature_selection.select_UniRep_features(UniRep_features)
    print(fastas)
    ASDC_features = get_ASDC(fastas)
    new_ASDC_feature = feature_selection.select_ASDC_features(ASDC_features)
    encoding.append(new_ASDC_feature)
    encoding.append(new_UniRep_features)
    feature = np.column_stack(encoding)
    
    print("Predicting......")
    scale = joblib.load('./models/scaler.pkl')
    scaled_feature = scale.transform(feature)
    model = joblib.load('./models/saved_model.pkl')
    y_pred_prob = model.predict_proba(scaled_feature)

    df_out = pd.DataFrame(np.zeros((y_pred_prob.shape[0], 3)),
                          columns=["Sequence_name", "Prediction", "probability"])
    y_pred = model.predict(feature)
    print(collections.Counter(y_pred))
    for i in range(y_pred.shape[0]):
        df_out.iloc[i, 0] = str(sequence_names[i])
        if y_pred[i] == 1:
            df_out.iloc[i, 1] = "Plant vacuole protein"
            df_out.iloc[i, 2] = "%.2f%%" % (y_pred_prob[i, 1] * 100)
        if y_pred[i] == -1:
            df_out.iloc[i, 1] = "Non-plant vacuole protein"
            df_out.iloc[i, 2] = "%.2f%%" % (y_pred_prob[i, 0] * 100)
    os.chdir(".\Results")
    df_out.to_csv(args.o,)
    print("Job finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",description="Identification of plant vacuole proteins by using deep representation learning features")
    parser.add_argument("-i", required=True, default=None, help="input fasta file")
    parser.add_argument("-o", default="Results.csv", help="output a CSV results file")
    args = parser.parse_args()

    time_start = time.time()
    print("Sequence checking......")
    sequence, sequence_names = read_fasta_sequences.read_protein_sequences(args.i)

    predict(sequence)

    time_end = time.time()
    print('Total time cost', time_end - time_start, 'seconds')


