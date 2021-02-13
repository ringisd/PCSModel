# pytorch mlp for binary classification
import csv
import sys
from numpy import vstack
from numpy import ndarray
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import LayerNorm
from torch.nn import ReLU
from torch.nn import GELU
from torch.nn import LeakyReLU
from torch.nn import Sigmoid
from torch.nn import Tanh
from torch.nn import Module
from torch.nn import Dropout
from torch import clamp
from torch.optim import SGD#
from torch.optim import Adam
from torch.optim import AdamW
from torch.nn import BCELoss
from torch.nn import MSELoss
from torch.nn import L1Loss
from torch.nn import SmoothL1Loss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import torch
import numpy


print("Libraries Loaded")


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        #once I get the npy files, I can comment from here
        df = read_csv('phd-work/Deep/encodeFeatRedoClean.csv', header=None, low_memory=False) #data file
        #df2 = read_csv('phd-work/Deep/KVALS.csv', header=None, low_memory=False) #data file
        
        # store the inputs and outputs
        self.X = df.values[:, 1:1513]
        self.y = df2.values[:,0]
        print("data initialised")
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        #sef.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
        
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.20):
        # determine sizes
        test_size = int(round(n_test * len(self.X)))
        train_size = int(len(self.X) - test_size)
        # calculate the split
        return random_split(self, [train_size, test_size])

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, n_inputs)            
        kaiming_uniform_(self.hidden1.weight, nonlinearity='leaky_relu')
        self.act1 = GELU() 
        self.norm1 = LayerNorm(n_inputs)
        self.hidden2 = Linear(n_inputs, 1000)            
        kaiming_uniform_(self.hidden2.weight, nonlinearity='leaky_relu')
        self.act2 = GELU() 

        self.hidden3 = Linear(1000, 800)            
        kaiming_uniform_(self.hidden3.weight, nonlinearity='leaky_relu')
        self.act3 = GELU()           

        self.hidden4 = Linear(800, 600)            
        kaiming_uniform_(self.hidden4.weight, nonlinearity='leaky_relu')
        self.act4 = GELU()

        self.hidden5 = Linear(600, 400)            
        kaiming_uniform_(self.hidden5.weight, nonlinearity='leaky_relu')
        self.act5 = GELU() 

        self.hidden6 = Linear(400, 200)            
        kaiming_uniform_(self.hidden6.weight, nonlinearity='leaky_relu')
        self.act6 = GELU() 

        self.hidden7 = Linear(200, 100)            
        kaiming_uniform_(self.hidden7.weight, nonlinearity='leaky_relu')
        self.act7 = GELU()   

        self.hidden8 = Linear(100, 64)            
        kaiming_uniform_(self.hidden8.weight, nonlinearity='leaky_relu')
        self.act8 = GELU() 

        self.hidden9 = Linear(64, 32)            
        kaiming_uniform_(self.hidden9.weight, nonlinearity='leaky_relu')
        self.act9 = GELU()

        self.hidden10 = Linear(32, 16)            
        kaiming_uniform_(self.hidden10.weight, nonlinearity='leaky_relu')
        self.act10 = GELU()

        self.hidden11 = Linear(16, 8)            
        kaiming_uniform_(self.hidden11.weight, nonlinearity='leaky_relu')
        self.act11 = GELU()

        self.hidden12 = Linear(8, 4)            
        kaiming_uniform_(self.hidden12.weight, nonlinearity='leaky_relu')
        self.act12 = GELU()                                         

        self.norm2 = LayerNorm(4)
        self.hiddenfinal = Linear(4, 1)
        #kaiming_uniform_(self.hiddenfinal.weight, nonlinearity='sigmoid')
        #self.actfinal =  Sigmoid()                    

        self.drop1 = Dropout(0.1)   

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.norm1(X)        
        X = self.drop1(X)

        X = self.hidden2(X)
        X = self.act2(X)
        X = self.drop1(X)


        X = self.hidden3(X)
        X = self.act3(X)
        X = self.drop1(X)
          

        X = self.hidden4(X)
        X = self.act4(X)

        X = self.hidden5(X)
        X = self.act5(X)

        X = self.hidden6(X)
        X = self.act6(X)

        X = self.hidden7(X)
        X = self.act7(X)

        X = self.hidden8(X)
        X = self.act8(X)
        X = self.drop1(X)
        
        X = self.hidden9(X)
        X = self.act9(X)                
        
        X = self.hidden10(X)
        X = self.act10(X)
        
        X = self.hidden11(X)
        X = self.act11(X)

        X = self.hidden12(X)
        X = self.act12(X)                
        
        X = self.norm2(X)
        X = self.hiddenfinal(X)
       # X = self.actfinal(X)

        return X

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    print ("Dataset loaded")
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True) #change batch size
    test_dl = DataLoader(test, batch_size=512, shuffle=False)
    print ("DataLoader done")
    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    # define the optimization
    print("training begin")
    criterion = L1Loss() #Check other  loss functions
    #optimizer = Adam(model.parameters(), lr=0.001, betas=(0.09, 0.999), eps=1e-08, weight_decay=0) # amsgrad=False)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)#check if other optimizaters work
    # enumerate epochs
    for epoch in range(300):
        # enumerate mini batches
        #print ('epoch',epoch)
        for i, (inputs, targets) in enumerate(train_dl):
            #to gpu
            #inputs, targets = inputs, targets
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            #print loss
            # update model weights
            optimizer.step()
            #print('epoch {}, loss {}'.format(epoch, loss.item()))
        print('epoch {}, loss {}'.format(epoch, loss.data))
        print(evaluate_model(test_dl,model))




# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals, mses = list(), list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        #inputs, targets = inputs.to(device), targets.to(device)
        # evaluate the model on the test set
        yhat = model(inputs)        
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    #print (actuals)
    #print (predictions) 
    acc = mean_absolute_error(actuals, predictions)#Do a calculation for "accuracy"
    return acc#mmse

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# prepare the data
path = 'phd-work/Deep/encodeFeatRedoClean.csv'
#plotter = VisdomLinePlotter(env_name='Tutorial Plots')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(1512)#self.X
                       
# train the model
train_model(train_dl, model)
# evaluate the model
finalMSE = evaluate_model(test_dl, model)
print('Validation MAE: %.6f' % finalMSE)
torch.save(model,'phd-work/Deep/model6.pth')

