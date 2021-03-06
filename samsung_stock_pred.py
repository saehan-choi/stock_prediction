import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import pandas_datareader as pdr
from sklearn.preprocessing import StandardScaler, MinMaxScaler

start = (2000, 1, 1)  # 2020년 01년 01월 
start = datetime.datetime(*start)  
end = datetime.date.today()  # 현재 


# yahoo 에서 삼성 전자 불러오기 
df = pdr.DataReader('005930.KS', 'yahoo', start, end)

X = df.drop(columns=['Volume','Adj Close'])
y = df.iloc[:, 5:6]

mm = MinMaxScaler()
ss = StandardScaler()
X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y) 
# y label은 정규화 시키지않고 해볼 것.
# 이분이 Adj Close를 예측하고 싶다하셨는데, Adj Close자체가 학습데이터로 들어감...
# 잘못된 부분같음.

# Train Data
X_train = X_ss[:4500, :]
X_test = X_ss[4500:, :]
# Test Data 
y_train = y_mm[:4500, :]
y_test = y_mm[4500:, :] 

X_train_tensors = torch.Tensor(X_train)
X_test_tensors = torch.Tensor(X_test)

y_train_tensors = torch.Tensor(y_train)
y_test_tensors = torch.Tensor(y_test)

X_train_tensors_final = X_train_tensors.unsqueeze(1)
X_test_tensors_final = X_test_tensors.unsqueeze(1)


# print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
#                         torch.Size([4500, 1, 4])     torch.Size([4500, 1])

# print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device


class LSTM1(nn.Module):
  def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
    super(LSTM1, self).__init__()
    self.num_classes = num_classes #number of classes
    self.num_layers = num_layers #number of layers
    self.input_size = input_size #input size
    self.hidden_size = hidden_size #hidden state
    # self.seq_length = seq_length #sequence length
 
    self.lstm = nn.LSTM(input_size, hidden_size,
                      num_layers, batch_first=True) #lstm
    self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
    self.fc = nn.Linear(128, num_classes) #fully connected last layer

    self.relu = nn.ReLU() 

  def forward(self,x):
    h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #hidden state
    c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #internal state   
    # Propagate input through LSTM

    output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
    # cn.shape == torch.Size([1, 4500, 2])
    # batch_first == True 일때 여기처럼 맨앞이 batch 갯수로 나오게됨

    print(hn.shape)
    # num_layers, x.size(0) (number of row), hidden_size

    hn = hn.squeeze(0) #reshaping the data for Dense layer next
    # hn.shape == torch.Size([4500, 2])
    out = self.relu(hn)
    out = self.fc_1(out) #first Dense
    # out.shape == torch.Size([4500, 128])
    out = self.relu(out) #relu
    out = self.fc(out) #Final Output
    # out.shape == torch.Size([4500, 1])
    return out 

num_epochs = 500 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 4 #number of features
hidden_size = 500 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]).to(device)

loss_function = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)  # adam optimizer



for epoch in range(num_epochs):
  outputs = lstm1.forward(X_train_tensors_final.to(device)) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
  loss = loss_function(outputs, y_train_tensors.to(device))

  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

df_X_ss = ss.transform(df.drop(columns=['Volume','Adj Close']))
df_y_mm = mm.transform(df.iloc[:, 5:6])

df_X_ss = torch.Tensor(df_X_ss) #converting to Tensors
df_y_mm = torch.Tensor(df_y_mm)
#reshaping the dataset

df_X_ss = df_X_ss.unsqueeze(1)

train_predict = lstm1(df_X_ss.to(device))#forward pass
data_predict = train_predict.data.detach().cpu().numpy() #numpy conversion
dataY_plot = df_y_mm.data.numpy()

# data_predict = mm.inverse_transform(data_predict) #reverse transformation
# dataY_plot = mm.inverse_transform(dataY_plot)

plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=4500, c='r', linestyle='--') #size of the training set

plt.plot(dataY_plot, label='Actuall Data') #actual plot
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show() 
