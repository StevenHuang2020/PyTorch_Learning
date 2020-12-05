#python3 Steven 10/02/20, Auckland,NZ
#pytorch study
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from backbones import *

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
        y = 0.8*X**3 - 1.8*X**2 + 0.1*X + 5.5 + noise

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

def preparDataSetMul(N=200, gamma=0.01): #multi features(variable) dataset
    """N: number of samples needed
    gamma: Noise coefficient
    """
    def noise(len):
        return np.random.rand(len)
    
    def fuc(x):
        return 2.2*x + 3.8
    
    def funN2(x1,x2):
        return 0.5*x1 + 0.5*x2 + 0.1
    
    def funN3(x1,x2,x3):
        return 2.2*x1 + 0.8*x2 + 1.7*x3 + 5

    if 0: #by torch
        #X = torch.unsqueeze(torch.linspace(-1, 3, N), dim=1)
        #y = X.pow(2) - 2.0*X + gamma*torch.rand(X.size()) 
        #X = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 4.0]])
        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5, 6]])
        #X = torch.tensor([1.0, 3.0])
        #y= torch.tensor([3.0, 7.0, 8])
        y= torch.tensor([3.0,4,5])
        #y = y.view(-1,2)
        
    else: #by numpy    
        #X0 = np.around(2*np.random.randn(N)+2, decimals=4)
        #X1 = np.around(2.3*np.random.randn(N)+1.6, decimals=4)
        X0 = np.random.rand(N)
        X1 = np.random.rand(N)
        
        y = funN2(X0,X1) #+ noise(N)*gamma
        #y = np.around(y,decimals=4)
        
        X0 = X0.reshape((N,1))
        X1 = X1.reshape((N,1))
        X = np.concatenate((X0, X1), axis=1)
        
        print('x0=', X0)
        print('x1=', X1)
        print('x=', X)
        print('y=', y)
        #numpy to torch object, method1
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        #X = torch.from_numpy(X)
        #y = torch.from_numpy(y)
        
    print('X.shape=', type(X), X.shape, X.dtype)
    print('y.shape=', type(y), y.shape, y.dtype)
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
    #optimizer = optim.AdamW(parameters, lr=lr)
    #optimizer = optim.SparseAdam(parameters, lr=lr)
    #optimizer = optim.ASGD(parameters, lr=lr)
    ##optimizer = optim.LBFGS(parameters, lr=lr)
    #optimizer = optim.Rprop(parameters, lr=lr)
    return optimizer

def optimizerDesc(optimizer):
    print('Optimizer defaults dict--------------------------')
    for key,value in optimizer.defaults.items():
        print('key:',key) #
        print(value)

    print('Optimizer state dict--------------------------')
    for key,value in optimizer.state.items():
        print('key:',key) #state param_groups
        print(value)
        
    print('Optimizer param groups------------------------')
    print('optimizer.param_groups:', len(optimizer.param_groups))#list
    #print(optimizer.param_groups)
    for i,param in  enumerate(optimizer.param_groups):
        print(f'\tOptimizer param groups {i}:')
        for key,value in param.items():
            print('\t\t key:',key) #params lr betas eps weight_decay
            print('\t\t',value)
                
    #lr = float(optimizer.param_groups[0]["lr"])
    #print('lr=',lr)
    
############################## Models start ######################################
class RegressionNet(nn.Module):
    def __init__(self, input=1, output=1, hidden=5):
        """Regression NN Model, three layers.
           input: input layer neuron numbers or feature numbers
           output: output layer neuron numbers, set to 1 when regression
           hidden: neuron numbers with one hidden layer
        """
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(input, hidden) #define fully connected layers
        self.fc2 = nn.Linear(hidden, hidden) #can add multiple layers
        self.fc3 = nn.Linear(hidden, output)
     
        #self.activeFun = None #linear model when not settting active function
        self.activeFun = F.elu #F.hardtanh #F.leaky_relu #F.relu #F.softsign #F.tanh   
        self.apply(self.weight_init) #customize weights initializer
        
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
    
    @torch.no_grad()
    def weight_init(self, m):
        #setting fixed parameters for compare optimizer
        #print('m=', m, 'className=', m.__class__.__name__)
        if type(m) == nn.Linear:
            m.weight.fill_(1.0)
            m.bias.fill_(1.0)
            #print('weight=', m.weight)
            #print('bias=', m.bias)
        else:
            #print('no linear m=', type(m), m)
            pass
                        
class RegressionNet2(nn.Module):
    def __init__(self, input=1, output=1, hidden=5, hiddenlayers=1):
        """Regression NN Model, multi-layers.
           input: input layer neuron numbers or feature numbers
           output: output layer neuron numbers, set to 1 when regression
           hidden: neuron numbers with one hidden layer
           hiddenlayers: number of hidden layers
        """
        super(RegressionNet2, self).__init__()
        self.fc1 = nn.Linear(input, hidden) #define fully connected layers
        
        self.fcs = []
        for i in range(hiddenlayers):
            self.fcs.append(nn.Linear(hidden, hidden))
        
        self.ouptput = nn.Linear(hidden, output)
     
        #self.activeFun = None #linear model when not settting active function
        self.activeFun = F.elu #F.hardtanh #F.leaky_relu #F.relu #F.softsign #F.tanh   
        #print('activeFun=',self.activeFun)
        
    def forward(self, x):
        x = self.activeFun(self.fc1(x))
        for fc in self.fcs:
            x = self.activeFun(fc(x))
            
        x = self.activeFun(self.ouptput(x))
        return x
    
class ClassifierNet(nn.Module):
    def __init__(self, input=1, output=1, hidden=5):
        """Fully connected classifier model.
        input: input layer neuron numbers or feature numbers
        output: output layer neuron numbers, class numbers
        """
        super(ClassifierNet, self).__init__()
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
    
class ClassifierCNN_Net(nn.Module):
    def __init__(self, output=2):
        """CNN classifier model.
        output: class numbers
        """
        super(ClassifierCNN_Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Start fc features: w*h*chn
        # Calculate: (w-f+2p)/s+1   (width - filter + 2* padding)/stride + 1
        # conv1: cnn: (28-3+0)/1+1 = 26
        # conv2: cnn: (26-3+0)/1+1 = 24, pooling: (24-2+0)/2 + 1 = 12
        
        self.fc1 = nn.Linear(12*12*64, 128)
        self.fc2 = nn.Linear(128, output)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
class ClassifierCNN_Net2(nn.Module):
    def __init__(self, output=2):
        """Sequential CNN classifier model.
        output: class numbers
        """
        super(ClassifierCNN_Net2, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
            ) #nn.Flatten()
        
        # Start fc features: w*h*chn
        # Calculate: (w-f+2p)/s+1   (width - filter + 2* padding)/stride + 1
        # conv1: cnn: (28-3+0)/1+1 = 26,  pooling: (26-3+0)/2 + 1 = 12
        # conv2: cnn: (12-3+0)/1+1 = 10,  pooling: (10-3+0)/2 + 1 = 4
        self.fc1 = nn.Linear(in_features=4*4*64,  out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        #x = x.view(-1, 5*5*64)
        #x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=1)
        return x
    
class ClassifierCNN_Net3(nn.Module):
    def __init__(self, output=2):
        """Sequential CNN classifier model.
        output: class numbers
        """
        super(ClassifierCNN_Net3, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            nn.Dropout(0.5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(8, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_features=4*4*16,  out_features=64),
            nn.Linear(in_features=64, out_features=output),
            #nn.Softmax(dim=1),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.net(x)

class ClassifierCNN_NetBB(nn.Module):
    def __init__(self, output=2, backbone=vgg16):
        """Sequential CNN classifier model.
        output: class numbers
        """
        super(ClassifierCNN_NetBB, self).__init__()
        
        self.net = backbone
        self.out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=1000,  out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_features=output),
            #nn.LogSoftmax(dim=1)
            nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        x = self.net(x)
        x = self.out(x)
        return x
    
#AdaptiveAvgPool2d
