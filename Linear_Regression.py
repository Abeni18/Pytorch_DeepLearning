# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:33:36 2018

@author: Abenezer
"""


import numpy as np


x_values = [i for i in range(11)]
#print(x_values)

x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1,1)  # column vector 
#print(x_train.shape)


# y = 2x +1 
y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
#print(y_train.shape)
y_train = y_train.reshape(-1,1)
#print(y_train.shape)


import torch
import torch.nn as nn
from torch.autograd import Variable 

'Step 1 - Create Model Class '

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        
        # single linear model -  x goes in y comes out 
        self.linear = nn.Linear(input_dim, output_dim)  # y = 2x + 1   input_dim=x   output_dim=y
        
    def forward(self, x):
        out = self.linear(x)
        return out
        
input_dim = 1
output_dim = 1

'Step 2 :- instanciate the model class'
model = LinearRegressionModel(input_dim, output_dim)

criterion = nn.MSELoss()   # MSE loss

'Step 3:- Instantiate Loss Class'
# instanciate optimizer :- this update the parameters (a and b) y = ax + b
learning_rate = 0.01

'Step 4:- Instantiate Optimizer Class'
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'Step 5:- training' 
epochs = 100

los = []
for epoch in range(epochs):
    epoch +=1

# convert numpy array to torch and to torch Variable
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))
    
# Clear gradients w.r.t parameters in every epochs to start from zero, no accumlation of gradient, to avoid gradient from previous epoch affect current one 
    optimizer.zero_grad()
    
    # Forward to get output
    outputs = model.forward(inputs)  

    # claculat loss
    loss = criterion(outputs, labels)
    los.append(loss.data[0])
    
    # Getting gradients W.r.t parameters 
    loss.backward()
    
    # Updating parameters
    optimizer.step()
    
    print('epochs {}, loss {}'.format(epochs, loss.data[0] ))
    
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
# Legend and plot
plt.legend(loc='best')


plt.figure(2)
x = [i for i in range(len(los))]
plt.plot(x, los)
plt.show()


'Save Model'
save_model = False
if save_model is True:
    # Save only parameters a and b
    torch.save(model.state_dict(), 'awesome_model.pkl')
    
'Load Model'
load_model = False
if load_model is True:
    model.load_state_dict(torch.load('awesome_model.pkl'))
    
    
