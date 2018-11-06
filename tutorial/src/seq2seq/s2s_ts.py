#!/usr/bin/env python
# encoding: utf-8
"""
@author: derek
@contact: derek.zheng8@gmail.com
@file: s2s_ts.py
@time: 2018/11/7 6:20
@desc:
"""
import torch
import pandas as pd
import numpy as np
import src.utils.utils as util
device = torch.device("cuda")
def prepara_data(data):
    data_input = torch.tensor(data[:,:7],dtype=torch.long, device=device)
    data_output = torch.tensor(data[:,7:],dtype=torch.long,device=device)
    return data_input,data_output

if __name__ == '__main__':
    data = util.gene_ts_array(1000,14)
    input,target = prepara_data(data)

