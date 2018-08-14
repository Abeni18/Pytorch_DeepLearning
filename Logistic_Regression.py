# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 22:54:14 2018

@author: Abenezer
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable 
import numpy as np
import matplotlib.pyplot as plt

'Step 1: Loading the dataset'

'loading training dataset'
train_dataset = dsets.MNIST(root = './data', train=True, transform=transforms.ToTensor(), download=True)
'loading test dataset'
test_dataset = dsets.MNIST(root = './data', train=False, transform=transforms.ToTensor())


'''
img = [train_dataset[i][0].numpy().reshape(28,28)  for i in range(3)]
#print(img)
label = train_dataset[2][1]
plt.imshow(img[0])

test_img = test_dataset[2][0].numpy().reshape(28,28) 
plt.imshow(test_img)
plt.show()  
'''


batch_size  = 100
n_iters = 3000

num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

'Step 2: Making data set iteratable: Training Dataset and Test Dataset'
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
'''#cheaking for iteratablity
import collections 
print(isinstance(train_loader, collections.Iterable))   #True if it is iteratable 
'''



'Step 3: Building Model'

'Linear Regression part'

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        out = self.linear(x)   # return y for the given x 
        return out
    
'Step 4: Instantiate Model Class'

input_dim = 28*28
output_dim = 10

model = LogisticRegressionModel(input_dim, output_dim)

'Step 5: Instantiate Loss Class'
criterion = nn.CrossEntropyLoss()  # this does both softmax and crossentropy

'Step 6: Instantiate Optimizer Class'
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'Step 7: Train the model'

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable 
        images = Variable(images.view(-1, 28*28))  #unlike linear regresion we don't have to change them to numpy
        labels = Variable(labels)
        
        # Clearing gradients w.r.t parameters 
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)
        
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        # getting gradients w.r.t parameters
        loss.backward()
        
        #updating parametes based on the gradients 
        optimizer.step()
        
        iter +=1
        
        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            
            # iterate through test dataset
            for images, labels in test_loader:
                # Load images to Torch Variable
                images = Variable(images.view(-1, 28*28))
                
                # Forward pass only to get logist/output
                outputs = model(images)
                
                # Get predictions from the maximum value 
                _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels 
                total += labels.size(0)
            
                # Total correct prediction
                correct += (predicted == labels).sum()
                
            accuracy = 100 * correct / total
            
            # Print loss
            print('Iteration: {} Loss: {} Accuracy: {}'.format(iter, loss.data[0], accuracy))
        
        
    


























