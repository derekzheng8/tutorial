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
import tutorial.src.utils.utils as util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

device = torch.device("cpu")


def prepara_data(data):
    data_input = torch.tensor(data[:, :7], dtype=torch.long, device=device)
    data_output = torch.tensor(data[:, 7:], dtype=torch.long, device=device)
    return data_input, data_output


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.nngru = nn.Linear(input_size, hidden_size)
        self.nngruout = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # print(input,hidden)
        output = self.nngru(input)
        output = self.nngruout(output)
        # output,hidden = self.gru(input,hidden)
        return output, hidden

    def init_hidden(self):
        return torch.rand(1, 1, self.hidden_size, device=device)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(1, hidden_size)
        self.o2o = nn.Linear(hidden_size, output_size)
        self.i2o = nn.Linear(1, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)
        # self.dropout = nn.Dropout(0.3)
        # self.batchnorm = nn.BatchNorm1d(hidden_size)

    def forward(self, input, hidden):
        # print(input)
        # print(hidden)
        # output,hidden = self.gru(input,hidden)
        # print(hidden)
        # print(list(hidden.size()))
        # hidden = self.batchnorm(hidden[0])
        # print(hidden)
        # hidden = self.dropout(hidden)
        # output1 = self.o2o(output)
        output2 = self.i2o(input)
        return output2, hidden


def train(encoder, decoder, criterion, encoder_optim, decoder_optim, input_tensor, target_tensor):
    Length = input_tensor.shape[0]
    Dim = input_tensor.shape[-1]
    encoder_outputs = torch.zeros(Length, 1, Dim, device=device)
    encoder_hidden = encoder.init_hidden()
    loss = 0
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    for i in range(Length):
        encoder_output, encoder_hidden = encoder(input_tensor[i].view(1, 1, -1), encoder_hidden)
    decoder_hidden = encoder_hidden
    # print(input_tensor[Length-1])
    decoder_input = input_tensor[Length - 1].view(1, 1, -1)
    for i in range(Length):
        # print(i)
        # print(decoder_input)
        # print("decoder_input",decoder_input)
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        # print("decoder_output",decoder_output)
        loss += criterion(target_tensor[i], decoder_output)
        # print(decoder_output)
        decoder_input = decoder_output.detach()
        # decoder_input = target_tensor[i].detach().view(1,1,-1)
        # print("another_input",decoder_input)
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), 0.1)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 0.1)
    encoder_optim.step()
    decoder_optim.step()
    return loss.item() / Length


def train_iter(inputs, targets, iters=100):
    # print(inputs,targets)
    encoder = Encoder(input_size=1, hidden_size=3, output_size=7).to(device)
    decoder = Decoder(hidden_size=3, output_size=1).to(device)
    encoder_optim = optim.ASGD(encoder.parameters(), lr=0.01)
    decoder_optim = optim.ASGD(decoder.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for i in range(iters):
        input = inputs[i].view(-1)
        target = targets[i].view(-1)
        # print(input,target)
        loss = train(encoder, decoder, criterion, encoder_optim, decoder_optim, input, target)
        if i % 10 == 0:
            print(i, loss)


if __name__ == '__main__':
    data = util.gene_ts_array(6000, 14)
    input, target = prepara_data(data)
    # test_input = torch.ones(1,1,1)
    # hidden_tensor = torch.zeros(1,1,7)
    # gru = nn.GRU(1,7)
    # output,hidden = gru(test_input,hidden_tensor)
    # output1 = F.softmax(output[0],dim=1)
    # topv, topi =output1.topk(1)
    # topv.unsqueeze(0).detach()
    input_tensor = torch.tensor(input, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
    train_iter(input_tensor, target_tensor, 5000)
