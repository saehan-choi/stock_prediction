import numpy as np
import pandas as pd
from torch import optim
import torch
from model import *
from utils import *


device = torch.device('cuda')

net = NeuralNet()
net.load_state_dict(torch.load('./weights/model_test.pt'))

# for name, param in net.named_parameters(): 
#     print(name, ':', param.requires_grad)

test_name = 'CTK'
test_path = f"./test/{test_name}_new_refined.csv"

# val_path = f"./input/{k}_validation.csv"
# test_path = f"./input/{k}_test.csv"

test_data = pd.read_csv(test_path)

test_data = numpy_to_tensor(test_data)

test_result = net(test_data)

test_result = torch.argmax(test_result, dim = 1)

print(test_result)

for i in range(len(test_result)):
    
    if test_result[i] == 0 or test_result[i] == 1 or test_result[i] == 4 or test_result[i] == 5:
        print(f'{i}번째 index입니다.')
