import numpy as np
import pandas as pd
from torch import optim
import torch
from model import *
from utils import *

# 테스트할 코인을 쓰시오
# coin = ['ATOM', 'BTC', 'ETH', 'GXC', 'HIBS', 'KLAY', 'LUNA', 'XRP']
# # , 


# for k in coin:

# train_path = f"./input/{k}_train.csv"
# val_path = f"./input/{k}_validation.csv"
# test_path = f"./input/{k}_test.csv"

train_path = f"./input/train.csv"
val_path = f"./input/validation.csv"
test_path = f"./input/test.csv"

train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path) 
test_data = pd.read_csv(test_path)


# rate,rate_after,g_up,up,middle,down,g_down

train_label_data = train_data[['g_up', 'up', 'l_up', 'l_down', 'down', 'g_down',]]
train_data = train_data.drop(['rate', 'rate_after', 'g_up', 'up', 'l_up', 'l_down', 'down', 'g_down'], axis=1)


val_label_data = val_data[['g_up', 'up', 'l_up', 'l_down', 'down', 'g_down']]
val_data = val_data.drop(['rate', 'rate_after', 'g_up', 'up', 'l_up', 'l_down', 'down', 'g_down'], axis=1)


# 데이터 나눌때 쓰세여 (train, validation, test)
# train_data = train_data.iloc[:]
# train_label_data = train_label_data.iloc[:]



train_data = numpy_to_tensor(train_data)
train_label_data = numpy_to_tensor(train_label_data)

val_data = numpy_to_tensor(val_data)
val_label_data = numpy_to_tensor(val_label_data)

batch_arr = [50, 60, 70, 80]
lr_arr = [1e-3, 1e-4, 1e-5]


net = NeuralNet()
criterion = nn.CrossEntropyLoss().cuda()
# optimizer = optim.Adam(net.parameters(),lr=1e-4)
# 이거 5로바꿔볼게
optimizer = optim.Adam(net.parameters(),lr=1e-4)
num_epochs = 50
batch_size = 80

batch_num_train = len(train_data) // batch_size
batch_num_val = len(val_data) // batch_size

for epochs in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    for i in range(batch_num_train):
        start = i * batch_size
        end = start + batch_size
        pred = train_data[start:end][:]
        label = train_label_data[start:end]
        # print(f'pred : {pred}')
        # print(f'pred shape:{pred.shape}')

        # print(f'label : {label}')
        # print(f'label shape:{label.shape}')


        # pred, label = pred.to(device), label.to(device)
        outputs = net(pred).squeeze()
        # print(outputs)

        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    with torch.no_grad():
        for j in range(batch_num_val):            
            start = j * batch_size
            end = start + batch_size

            pred_val = val_data[start:end][:]
            label_val = val_label_data[start:end]
            outputs_val = net(pred_val).squeeze()
            val_loss = criterion(outputs_val, label_val)
            val_loss += val_loss.item()






    print(f'epochs : {epochs}')
    print(f'train_loss : {train_loss/batch_num_train}')
    print(f'val_loss : {val_loss/batch_num_val}')


PATH = './weights/'
torch.save(net.state_dict(), PATH+f'model_test.pt')


