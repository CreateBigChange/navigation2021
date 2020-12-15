# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch.utils.data as Data
import torch
from torch import nn



if __name__ == '__main__':

    print(1)







    # Hyper Parameters
    TIME_STEP = 7  # rnn time step
    INPUT_SIZE = 61  # rnn input size
    LR = 0.02  # learning rate
    BATCH_SIZE = 8








    #label WIFI、BLE、MAG data


    # #laod data  to loader
    #
    # torch_dataset = Data.TensorDataset(torch.from_numpy(x_train).to(torch.float32),
    #                                    torch.from_numpy(y_train).to(torch.int64))
    #
    # loader = Data.DataLoader(
    #     dataset=torch_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #
    # )












# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()
#
#         self.rnn = nn.RNN(
#             input_size=INPUT_SIZE,
#             hidden_size=64,  # rnn hidden unit
#             num_layers=2,  # number of rnn layer
#             batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
#         )
#         self.out = nn.Linear(64, 5)
#
#     def forward(self, x, h_state):
#         # x (batch, time_step, input_size)
#         # h_state (n_layers, batch, hidden_size)
#         # r_out (batch, time_step, hidden_size)
#         r_out, h_state = self.rnn(x, h_state)
#
#         # outs = []  # save all predictions
#         # for time_step in range(r_out.size(1)):  # calculate output for each time step
#         #     outs.append(self.out(r_out[:, time_step, :]))
#
#         out = self.out(r_out[:, -1, :])
#         return out, h_state
#
#         # instead, for simplicity, you can replace above codes by follows
#         # r_out = r_out.view(-1, 32)
#         # outs = self.out(r_out)
#         # outs = outs.view(-1, TIME_STEP, 1)
#         # return outs, h_state
#
#         # or even simpler, since nn.Linear can accept inputs of any dimension
#         # and returns outputs with same dimension except for the last
#         # outs = self.out(r_out)
#         # return outs
#
#
#
#
# rnn = RNN()
# print(rnn)
#
# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
# loss_func = torch.nn.CrossEntropyLoss()
#
# h_state = None  # for initial hidden state
#
# # plt.figure(1, figsize=(12, 5))
# # plt.ion()  # continuously plot
# for epoch in range(100):
#     accuracy = 0
#     for step,(batch_x,batch_y) in enumerate(loader):
#         b_x = batch_x.view(BATCH_SIZE, TIME_STEP, INPUT_SIZE)  # batch_size time_step, input_size
#
#
#
#         prediction, h_state = rnn(b_x, h_state)  # rnn output
#         # !! next step is important !!
#         h_state = h_state.data  # repack the hidden state, break the connection from last iteration
#
#         loss = loss_func(prediction, batch_y)  # calculate loss
#         optimizer.zero_grad()  # clear gradients for this training step
#         loss.backward()  # backpropagation, compute gradients
#         optimizer.step()  # apply gradients
#
#
#
#
#         print(loss)

