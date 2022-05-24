#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Time : 2022.05.18
# @Author : jiaoshihu
# @Email : shihujiao@163.com
# @IDE : PyCharm
# @File : iPVP_DRLP.py

def get_DPC(fastas):
    import re
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        sequence = re.sub('-', '', i)
        code = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings