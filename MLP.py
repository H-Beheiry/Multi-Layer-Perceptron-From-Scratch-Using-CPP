# This script is intentionally unoptimized to replicate the exact architecture 
# and execution flow of my C++ implementation for benchmarking purposes.
#
# CONSTRAINTS:
# 1. Single-threaded CPU execution (torch.set_num_threads(1))
# 2. Batch processing (iterating one sample at a time)
# 3. No DataLoader (manual indexing)
# 4. Stochastic Gradient Descent (batch_size=1)

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time

torch.set_num_threads(1)
INPUT_SIZE= 784
HIDDEN_SIZE= 64
OUTPUT_SIZE= 10
EPOCHS= 1
TRAIN_SIZE= 25000
TEST_SIZE= 5000
LEARNING_RATE= 0.15
DEVICE= "cpu"

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1= nn.Linear(INPUT_SIZE,HIDDEN_SIZE)
        self.sigmoid= nn.Sigmoid()
        self.L2=nn.Linear(HIDDEN_SIZE,OUTPUT_SIZE)
    def forward(self,x):
        out= self.L1(x)
        out= self.sigmoid(out)
        out= self.L2(out)
        return self.sigmoid(out)

def load_data(data,size):
    print(f"loading {size} samples from {data}")
    df= pd.read_csv(data, nrows=size, header=None, skiprows=1)
    labels= torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
    pixels= torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32) / 255.0
    return pixels, labels

def train_test_loop(model,loss_fn,optm,train_data_x,train_data_y,test_data_x,test_data_y):
    train_start_time= time.time()
    for i in range(len(train_data_x)):
        optm.zero_grad()
        output= model(train_data_x[i])
        loss= loss_fn(output,train_data_y[i])
        loss.backward()
        optm.step()
        if i%200==0:
            print(f"ITER: {i} |  LOSS {loss}")
    train_end_time= time.time()
    print(f"TRAIN TIME= {train_end_time-train_start_time:.4f}")

    test_start_time= time.time()
    with torch.no_grad():
        output= model(test_data_x)
        predicted = torch.argmax(output, dim=1)
        correct = (predicted == test_data_y).sum().item()
    test_end_time = time.time()
    print(f"ACCURACY: {100 * correct / len(test_data_y)}")
    print(f"TEST TIME: {test_end_time-test_start_time}")

if __name__=="__main__":
    full_pixels, full_labels= load_data("mnist_train.csv", TRAIN_SIZE + TEST_SIZE)
    train_x= full_pixels[:TRAIN_SIZE]
    train_y= full_labels[:TRAIN_SIZE]
    test_x = full_pixels[TRAIN_SIZE:]
    test_y = full_labels[TRAIN_SIZE:]
    train_y_onehot = torch.nn.functional.one_hot(train_y, num_classes=10).float()
    model= MLP().to(DEVICE)
    loss_fn= nn.MSELoss()
    optm= optim.SGD(model.parameters(), lr=LEARNING_RATE,)
    train_test_loop(model,
                    loss_fn,
                    optm,
                    train_x,
                    train_y_onehot,
                    test_x,
                    test_y)