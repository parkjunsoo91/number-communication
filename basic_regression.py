import torch
from torch.utils.data import Dataset, DataLoader, random_split
import math
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 16, 1, 100, 1

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in) * 100
y = x

class MyDataset(Dataset):
    def __init__(self, x_list):
        self.continuous_data(x_list)

    def continuous_data(self, x_list):
        #continuous data
        #we are given x_list in python list format, and create x, y pairs
        self.x = torch.tensor(x).to(torch.float)
        self.y = torch.tensor(x).to(torch.float)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        #should return numpy array
        return {'x': self.x[idx], 'y': self.y[idx]}


##### train data and test data section #######
# what is going to be trained and what is going to be tested??
NUM_DATA = 10
TEST_SIZE = int(NUM_DATA/10)*1
raw_data = [i for i in range(NUM_DATA)]
raw_data = [math.pow(1.2,i) for i in range(50)]
raw_data = np.random.uniform(0,100,100)
#x = torch.tensor(raw_data)
#y = torch.tensor(raw_data)
dataset1 = MyDataset(x)
dataset2 = MyDataset(y)
train_dataloader = DataLoader(dataset1, batch_size = 1)
test_dataloader = DataLoader(dataset2, batch_size = 100)
print(x)
print(dataset1.x)
print(dataset1[:]['x'])
for i, s in enumerate(train_dataloader):
    print(s['x'])
# Use the nn package to define our model and loss function.

model = torch.nn.Linear(D_in, D_out)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e+2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # y_pred = model(x)
    # loss = loss_fn(y_pred, y)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    
    for i_batch, sample_batched in enumerate(train_dataloader):
        print("haha")
        print(sample_batched['x'])
        print(sample_batched['x'].size())
        print(sample_batched['y'])
        print(sample_batched['y'].size())
        exit()
        y_pred = model(sample_batched['x'])

        if t == 499:
            for i in range(N):
                print(sample_batched['x'][i].item(), sample_batched['y'][i].item(), y_pred[i][0].item())

    # Compute and print loss.
        loss = loss_fn(y_pred, sample_batched['y'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(t, loss.item())
    for param in model.parameters():
        print(param.data)

