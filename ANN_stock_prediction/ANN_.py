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

train_path = f"./input/train_c.csv"
val_path = f"./input/validation_c.csv"
test_path = f"./input/test_c.csv"

train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path) 
test_data = pd.read_csv(test_path)




train_data = train_data.sample(frac=1)
# 데이터 섞기!!!

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

batch_arr = [20, 30, 40, 50, 60]
lr_arr = [1e-3, 1e-4, 5e-4, 1e-5]


for l in batch_arr:
    f = open("C:/Users/Administrator/Desktop/stock_predic/stock_prediction/ANN_stock_prediction/result.txt", 'a')
    for q in lr_arr:
        net = NeuralNet()
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(net.parameters(),lr=1e-4)
        # 이거 5로바꿔볼게
        optimizer = optim.Adam(net.parameters(),lr=q)
        num_epochs = 20
        batch_size = l

        batch_num_train = len(train_data) // batch_size
        batch_num_val = len(val_data) // batch_size

        print(f'train:{batch_num_train}')
        print(f'val:{batch_num_val}')

        for epochs in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            for i in range(batch_num_train):
                start = i * batch_size
                end = start + batch_size
                pred = train_data[start:end][:]
                label = train_label_data[start:end]
                # pred, label = pred.to(device), label.to(device)
                outputs = net(pred).squeeze()

                loss = criterion(outputs, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            net.eval()
            with torch.no_grad():
                for j in range(batch_num_val):            
                    start = j * batch_size
                    end = start + batch_size
                    pred_val = val_data[start:end][:]
                    label_val = val_label_data[start:end]
                    outputs_val = net(pred_val).squeeze()
                    loss = criterion(outputs_val, label_val)
                    # 이거 실수 주의하기
                    val_loss += loss.item()

            print(f'epochs : {epochs}')
            print(f'train_loss : {train_loss/batch_num_train}')
            print(f'val_loss : {val_loss/batch_num_val}')
            print(f'batch_size:{batch_size} learning_rate{q}')


            f.write(f'epochs : {epochs}\n')
            f.write(f'train_loss : {train_loss/batch_num_train}\n')
            f.write(f'val_loss : {val_loss/batch_num_val}\n')
            f.write(f'batch_size:{batch_size} learning_rate{q}\n')

        

        PATH = './weights/'
        torch.save(net.state_dict(), PATH+f'batch{l}_lr{q}_model_test.pt')
    f.close()

