import torch
import torch.nn as nn


input = torch.randn(4, 7, 5)

rnn_layer = nn.RNN(input_size=5, hidden_size=4, num_layers=3)
print(rnn_layer)

# # 결과
# # RNN(5, 4, num_layers=3, batch_first=True)
(output, hidden) = rnn_layer(input)


print("Output size : {}".format(output.size()))
print("Hidden size : {}".format(hidden.size()))

print(output[-1])

print(hidden)
# # output : [sequence, batch_size, hidden_size] 
# # hidden : [num_layer, batch_size, hidden_size] 

