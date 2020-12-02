#python3 Steven 10/02/20, Auckland,NZ
#pytorch study
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def preparDataSet(N=200, gamma=0.01): #1 feature(variable) dataset
    """N: number of samples needed
    gamma: Noise coefficient
    """
    if 1: #by torch
        X = torch.unsqueeze(torch.linspace(-1, 3, N), dim=1)
        y = X.pow(2) - 2.0*X + gamma*torch.rand(X.size()) 
    else: #by numpy
        X = np.linspace(-1, 3, N)
        noise = np.random.randn(X.shape[0]) * gamma
        y = X**3 - 2.0*X**2 + 1.8*X + 5.5 + noise

        X = X.reshape((N,1))
        y = y.reshape((N,1))
        
        #numpy to torch object, method1
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        
        #numpy to torch object, method2
        # X = X.astype(np.float32)
        # y = y.astype(np.float32)
        # X = torch.from_numpy(X)
        # y = torch.from_numpy(y)
        
    print('X.shape=',X.shape)
    print('y.shape=',y.shape)
    return X, y

def lossFunction(): #https://pytorch.org/docs/stable/nn.html#loss-functions
    # f = nn.L1Loss() #mean absolute error
    f = nn.MSELoss() #mean suqare error
    # f = nn.CrossEntropyLoss()
    # f = nn.CTCLoss()
    # f = nn.NLLLoss()
    # f = nn.PoissonNLLLoss()
    # f = nn.KLDivLoss()
    # f = nn.BCELoss()
    # f = nn.BCEWithLogitsLoss()
    # f = nn.MarginRankingLoss()
    # f = nn.HingeEmbeddingLoss()
    # f = nn.MultiLabelMarginLoss()
    # f = nn.SmoothL1Loss()
    # f = nn.SoftMarginLoss()
    # f = nn.MultiLabelSoftMarginLoss()
    # f = nn.CosineEmbeddingLoss()
    # f = nn.MultiMarginLoss()
    # f = nn.TripletMarginLoss()
    # f = nn.TripletMarginWithDistanceLoss()

    return f
    
def optimizerTorch(parameters, lr=1e-3):
    #optimizer = optim.RMSprop(parameters, lr=lr)
    optimizer = optim.Adamax(parameters, lr=lr)
    #optimizer = optim.Adam(parameters, lr=lr)
    #optimizer = optim.SGD(parameters, lr=lr)
    #optimizer = optim.Adadelta(parameters, lr=lr)
    #optimizer = optim.Adagrad(parameters, lr=lr)
    #optimizer = optim.Adadelta(parameters, lr=lr)
    #optimizer = optim.AdamW(parameters, lr=lr)
    #optimizer = optim.SparseAdam(parameters, lr=lr)
    #optimizer = optim.ASGD(parameters, lr=lr)
    ##optimizer = optim.LBFGS(parameters, lr=lr)
    #optimizer = optim.Rprop(parameters, lr=lr)
    return optimizer

class RegressionNet(nn.Module):
    def __init__(self, input=1, output=1, hidden=5):
        """input: input layer neuron numbers or feature numbers
           output: output layer neuron numbers, set to 1 when regression
        """
        super().__init__()
        self.fc1 = nn.Linear(input, hidden) #define fully connected layers
        self.fc2 = nn.Linear(hidden, hidden) #can add multiple layers
        self.fc3 = nn.Linear(hidden, output)
     
        #self.activeFun = None #linear model when not settting active function
        self.activeFun = F.elu #F.hardtanh #F.leaky_relu #F.relu #F.softsign #F.tanh   
        #print('activeFun=',self.activeFun)
        
    def forward(self, x):
        if self.activeFun:
            x = self.activeFun(self.fc1(x))
            x = self.activeFun(self.fc2(x))
            x = self.activeFun(self.fc3(x))
            #x = F.hardtanh(self.fc4(x),-2,5)
        else:
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        return x
    
class ClassifierNet(nn.Module):
    def __init__(self, input=1, output=1, hidden=5):
        """input: input layer neuron numbers or feature numbers
           output: output layer neuron numbers, class numbers
        """
        super().__init__()
        self.fc1 = nn.Linear(input, hidden) #define fully connected layers
        self.fc2 = nn.Linear(hidden, hidden) #can add multiple layers
        self.fc3 = nn.Linear(hidden, output)
     
        #self.activeFun = None #linear model when not settting active function
        self.activeFun = F.relu #F.elu #F.hardtanh #F.leaky_relu #F.softsign #F.tanh   
        #print('activeFun=',self.activeFun)
        
    def forward(self, x):
        x = self.activeFun(self.fc1(x))
        x = self.activeFun(self.fc2(x))
        x = self.activeFun(self.fc3(x))
        #x = F.hardtanh(self.fc3(x),-2,5)
        
        x = F.softmax(x,dim=1)
        return x