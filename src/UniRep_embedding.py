#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time : 2022.05.18
# @Author : jiaoshihu
# @Email : shihujiao@163.com
# @IDE : PyCharm
# @File : iPVP_DRLP.py

from __future__ import print_function,division
import sys
import os
sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
import time
from rich.progress import track
import numpy as np
import pandas as pd
import torch
import warnings
from tape import UniRepModel,TAPETokenizer
warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def UniRep_Embed(fastaFile):
    UNIREPEB_=[]
    model = UniRepModel.from_pretrained('babbler-1900')
    model=model.to(DEVICE)
    tokenizer = TAPETokenizer(vocab='unirep')
    for seq in track(fastaFile,"Computing: "):
        sequence = seq
        with torch.no_grad():
            token_ids = torch.tensor([tokenizer.encode(sequence)])
            token_ids = token_ids.to(DEVICE)
            output = model(token_ids)
            unirep_output = output[0]
            #print(unirep_output.shape)
            unirep_output=torch.squeeze(unirep_output)
            #print(unirep_output.shape)
            unirep_output= unirep_output.mean(0)
            unirep_output = unirep_output.cpu().numpy()
           # print(sequence,len(sequence),unirep_output.shape)
            UNIREPEB_.append(unirep_output.tolist())
    unirep_feature=pd.DataFrame(UNIREPEB_)
    return unirep_feature
    