#!/usr/bin/env python
# encoding: utf-8
"""
@author: derek
@contact: derek.zheng8@gmail.com
@file: testing.py
@time: 2018/11/5 22:16
@desc:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,output_size)

    def forward(self, input,hidden):
        embedding = self.embedding(input)
        output,hidden = self.gru(embedding,hidden)
        return output,hidden
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)

class Decoder(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size,hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input,hidden):
        output,hidden = self.gru(input,hidden)
        output = self.softmax(output)
        return output,hidden

class attendDecoder(nn.Module):
    def __init__(self,hidden_size,output_size,dropout_p):
        super(attendDecoder,self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size,hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.softmax = nn.LogSoftmax(dim=1)
        self.atten = nn.Linear(hidden_size*2,output_size)
        self.i2h = nn.Linear(hidden_size *2,hidden_size)
        self.gru = nn.GRU(output_size,hidden_size)
        self.out = nn.Linear(hidden_size,output_size)

    def forward(self, input,hidden,encoder_outputs):
        embed = self.embedding(input).view(1, 1, -1)
        embed = self.dropout(embed)
        com_input_out = torch.cat((embed[0],hidden[0]),1)
        atten_weights = self.softmax(self.atten(com_input_out))
        atten_applied = torch.bmm(atten_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))
        output = torch.cat((embed[0],atten_applied[0]),1)
        output = self.i2h(output)
        output = F.relu(output)
        output,hidden = self.gru(output,hidden)
        output = F.log_softmax(self.out(output[0]),dim=1)
        return output,hidden,atten_weights
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
