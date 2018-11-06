#!/usr/bin/env python
# encoding: utf-8
"""
@author: derek
@contact: derek.zheng8@gmail.com
@file: testing.py
@time: 2018/11/5 5:50
@desc:
"""
import torch
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Linear(self.input_size + 128 + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(self.input_size + 128 + self.hidden_size, self.output_size)
        self.o2o = nn.Linear(self.hidden_size + self.output_size,self.output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(self.hidden_size,self.output_size)
    def initHidden(self):
        return torch.zeros(1,self.hidden_size)
    def forward(self, category,input,hidden):
        com = torch.cat((category,input,hidden),1)
        hidden = self.i2h(com)
        output = self.i2o(com)
        com_out = torch.cat(hidden,output)
        output = self.o2o(com_out)
        output = self.dropout(output)
        output = self.softmax(output)

