import numpy as np
import pandas as pd
from py import test
from torch import optim
import torch
from model import *
from utils import *


# device = torch.device('cuda')

net = NeuralNet()
net.load_state_dict(torch.load('./weights/batch60_lr0.0005_model_test.pt'))

# batch40_lr1e-05_model_test 

# for name, param in net.named_parameters(): 
#     print(name, ':', param.requires_grad)

test_name = 'CTC'
test_path = f"./test/{test_name}_new.csv"



accuracy_arr = []


test_data = pd.read_csv(test_path)
test_data_val = test_data[['g_up', 'up', 'l_up', 'l_down', 'down', 'g_down']].copy()
test_data = test_data.drop(columns= ['rate', 'rate_after', 'g_up', 'up', 'l_up', 'l_down', 'down', 'g_down'])


for i in range(len(test_data_val)):
    accuracy_arr.append(test_data_val.loc[i].argmax())

# argmax를 통해 값 가져옴
# test_data_val.loc[0] -> 0번째 행 데이터 가져옴   그다음 argmax 를 통해 최댓값 가져옴
# ex) g_up      0.0
    # up        0.0
    # l_up      0.0
    # l_down    0.0
    # down      0.0
    # g_down    1.0



test_data = numpy_to_tensor(test_data)
test_result = net(test_data)
test_result = torch.argmax(test_result, dim = 1)

k = 0

up = 0
down = 0

real_up = 0
real_down = 0



for i in range(len(test_result)):
    if int(test_result[i]) == accuracy_arr[i]:
        # print('맞았습니다')
        k+=1

    if int(accuracy_arr[i])==0 or int(accuracy_arr[i])==1 or int(accuracy_arr[i])==2:
        real_up +=1
        if int(test_result[i])==0 or int(test_result[i])==1 or int(test_result[i])==2:
            up +=1

    
    elif int(accuracy_arr[i])==3 or int(accuracy_arr[i])==4 or int(accuracy_arr[i])==5:
        real_down +=1
        if int(test_result[i])==3 or int(test_result[i])==4 or int(test_result[i])==5:
            down +=1


print(f'{len(test_result)}개중에서 {k}개 맞았습니다')
print(f'{k/len(test_result)*100}% 확률로 정답입니다')

print('\n')
# print(f'pred_up의 갯수는 : {up}')
# print(f'pred_down의 갯수는 : {down}')

# print(f'real_up의 갯수는 : {real_up}')
# print(f'real_down의 갯수는 : {real_down}')


print(f'up 추세{up/real_up*100}% 확률로 맞췄습니다')
print(f'down 추세{down/real_down*100}% 확률로 맞췄습니다')





