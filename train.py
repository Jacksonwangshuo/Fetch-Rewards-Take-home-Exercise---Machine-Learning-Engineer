#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

df = pd.read_csv('data_daily.csv')

date = df['# Date'].values.tolist()
receipt_count = df['Receipt_Count'].values.tolist()

X, y = [], []
for i in range(7, len(receipt_count)):
    x = receipt_count[i - 7: i]
    X.append(x)
    y.append(receipt_count[i])
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X = X / 20000000
y = y / 20000000


class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(in_features=7, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

lr = LR()

X = torch.from_numpy(X).to(torch.float32)
y = torch.from_numpy(y).to(torch.float32)


loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(lr.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()

min_loss = float('inf')
for epoch in range(1000):
    y_pred = lr(X)
    loss = loss_fn(y_pred, y)  # calculate loss
    loss.backward()  # loss backward
    optimizer.step()  # update weights

    flag = ''
    if loss.item() < min_loss:
        torch.save(lr, 'lr.pt')
        min_loss = loss.item()
        flag = '*'
    print(f'Epoch: {epoch + 1:4d}\tLoss: {loss.item():.4f}\t{flag}')

lr = torch.load('lr.pt')
lr.eval()

receipt_count_2022 = receipt_count[-7:] + [0 for _ in range(len(receipt_count))]
receipt_count_2022 = [num / 20000000 for num in receipt_count_2022]

for i in range(7, len(receipt_count_2022)):
    x = [receipt_count_2022[i - 7: i]]
    x = np.array(x, dtype=np.float32)
    x = torch.from_numpy(x).to(torch.float32)
    with torch.no_grad():
        y_pred = lr(x).detach().numpy().tolist()[0][0]
    receipt_count_2022[i] = y_pred

col_names = ['# Date', 'Receipt_Count']

data = []
for i, j in zip(date, receipt_count_2022[7:]):
    data.append([i, j])
# change to DataFrame
df_2022 = pd.DataFrame(data, columns=col_names)


def change_date(string):
    return string.replace('2021', '2022')


def change_recp(number):
    return int(number * 20000000)


df_2022['# Date'] = df_2022['# Date'].apply(change_date)
df_2022['Receipt_Count'] = df_2022['Receipt_Count'].apply(change_recp)
df_2022.to_csv('data_daily_2022.csv', index=False)
