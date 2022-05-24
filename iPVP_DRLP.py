#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time : 2022.05.18
# @Author : jiaoshihu
# @Email : shihujiao@163.com
# @IDE : PyCharm
# @File : iPVP_DRLP.py

import numpy as np
import pandas as pd
import joblib
import argparse
import time
import os
import re
import warnings
from src.features import get_feature

warnings.filterwarnings('ignore')

if not os.path.exists("results"):
    os.makedirs("results")



def read_protein_sequences(file):
    with open(file) as f:
        data = f.read()
    if re.search('>', data) == None:
        print("Please input correct FASTA format protein sequence！！！")
    else:
        records = data.split('>')[1:]
        sequences = []
        sequence_name = []
        for fasta in records:
            array = fasta.split('\n')
            header, sequence = array[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
            name = header
            sequences.append(sequence)
            sequence_name.append(name)
        return sequences, sequence_name

def predict(seqs):
    final_feature = get_feature(seqs)
    scale = joblib.load('./models/scaler.pkl')
    scaled_UniRep_features = scale.transform(final_feature)
    model = joblib.load('./models/saved_LGBM_model.pkl')
    y_pred_prob = model.predict_proba(scaled_UniRep_features)

    df_out = pd.DataFrame(np.zeros((y_pred_prob.shape[0], 3)),columns=["Sequence_name", "Prediction", "probability"])
    y_pred = model.predict(final_feature)
    for i in range(y_pred.shape[0]):
        df_out.iloc[i, 0] = str(sequence_names[i])
        if y_pred_prob[i, 1] >= 0.5:
            df_out.iloc[i, 1] = "Vacuolar Protein"
            df_out.iloc[i, 2] = "%.2f%%" % (y_pred_prob[i, 1] * 100)
        else:
            df_out.iloc[i, 1] = "Non Vacuolar Protein"
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
    print("Work launched......")
    sequences, sequence_names = read_protein_sequences(args.i)
    predict(sequences)
    time_end = time.time()
    print('Total time cost', time_end - time_start, 'seconds')


