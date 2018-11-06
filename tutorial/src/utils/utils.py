#!/usr/bin/env python
# encoding: utf-8
"""
@author: derek
@contact: derek.zheng8@gmail.com
@file: utils.py
@time: 2018/11/7 6:22
@desc:
"""
import pandas as pd
import numpy as np

def gene_ts_data(length):
    a = np.random.randint(1,100+1)
    data = np.arange(a,a+length)
    return data
def gene_ts_array(size,length):
    return np.array([gene_ts_data(length) for _ in range(size)])

if __name__ == '__main__':
    a = gene_ts_array(5,10)