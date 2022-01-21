import numpy as np
import pandas as pd
from torch import optim
import torch
from model import *
from utils import *

train_path = "./input/train.csv"
test_path = "./input/test.csv"
sample_sub = "./input/sample_submission.csv"

train_data = pd.read_csv(train_path)
val_data = pd.read_csv(test_path) 
test_data = pd.read_csv(sample_sub) 


train_label_data = train_data['pressure']
train_data = train_data.drop(['id','breath_id','pressure'], axis=1)
# 여기 R하고 C 도 drop 해야함. R_5 R_20 R_50 이미있음.

print(train_data)

train_data = train_data.iloc[:]
train_label_data = train_label_data.iloc[:]

train_data = numpy_to_tensor(train_data)
train_label_data = numpy_to_tensor(train_label_data)

device = torch.device('cuda')
net = NeuralNet()
net = net.to(device)


criterion = nn.L1Loss().cuda()
# in this competition loss function is MAE Loss so I used it.
optimizer = optim.Adam(net.parameters(),lr=1e-3)
num_epochs = 25
batch_size = 80
batch_num_train = len(train_data) // batch_size
batch_num_val = len(val_data) // batch_size

for epochs in range(num_epochs):
    train_loss = 0.0
    test_loss = 0.0
    for i in range(batch_num_train):
        start = i * batch_size
        end = start + batch_size
        pred = train_data[start:end][:]
        label = train_label_data[start:end]
        pred, label = pred.to(device), label.to(device)
        outputs = net(pred).squeeze()
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'epochs : {epochs}')
    print(f'train_loss : {train_loss/batch_num_train}')
    
PATH = './weights/'
torch.save(net.state_dict(), PATH+'model_alot_of_feature.pt')


