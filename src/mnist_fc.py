#python3 Steven 12/01/20,Auckland,NZ
#pytorch study
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os,time
from commonTorch import ClassifierNet,optimizerTorch

def testLayers():
    input = torch.randn(3, 2)
    #m = nn.Softmax(dim=1)
    #output = m(input)
    output1 = F.softmax(input, dim=1)
    output2 = F.log_softmax(input, dim=1)
    output3 = F.sigmoid(input)
    print('input=',input)
    print('output1,softmax=',output1)
    print('output2,log_softmax=',output2)
    print('output3,sigmoid=',output3)
    
def descpritDataset(dataset):
    for data in dataset:
        print(len(data), type(data), data[0][0].shape) #data
        x,y = data[0][0],data[1][0]
        print('y=',y)
        plt.imshow(x.view([28,28,1]))
        plt.show()
        break
    
    total = 0
    counter_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    for data in dataset:
        Xs,ys = data
        for y in ys:
            counter_dict[int(y)] += 1
            total += 1
            
    print(counter_dict,'total=',total)
    for i in counter_dict:
        print(f"{i}: {round(counter_dict[i]*100/total,4)}%")
    
def prepareData(batch_size=10):
    train = datasets.MNIST(r"./res/", train=True,download=False,
                       transform=transforms.Compose([transforms.ToTensor()])) #first run download=True
    test = datasets.MNIST(r"./res/", train=False,
                        transform=transforms.Compose([transforms.ToTensor()]))

    trainset = torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=True)
    testset = torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=True)

    print(type(train), train)
    print('train.targets=', train.targets)
    #print(type(test),test)
    print(type(trainset),trainset)
    # print(type(testset),testset)
    
    #descpritDataset(trainset)
    return trainset,testset,train,test

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        if 1:
            x = self.fc1(x) 
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.fc4(x) 
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
        
        x = F.softmax(x,dim=1)    
        return x
    
def accuracy(net,dataset):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset:
            X,y = data
            output = net(X.view(-1,28*28))
            for idx, i in enumerate(output):
                #if torch.argmax(i) == y[idx]:
                    #correct += 1
                correct = correct + int(torch.argmax(i) == y[idx])
                total += 1
    print('Accuracy: ', round(correct/total,3))

def saveModel(net, optimizer, epoch, loss, save_dir):
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
        net_path = os.path.join(save_dir, 'train_e%d.pth' % epoch)
        
        #print('dict=',self.net.state_dict())
        # for key in self.net.state_dict():
        #     print('key=', key)
        
        torch.save({
        'epoch': epoch+1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, net_path)
        
        #torch.save(self.net.state_dict(), net_path)
        
def writeLog(log):
        logFile=r'log'
        if not os.path.exists(logFile):
            os.makedirs(logFile)
        
        logFile=logFile + '/' + 'log.txt'
        with open(logFile,'a',newline='\n') as dstF:
            dstF.write(log)
               
def main():
    #testLayers()
    trainset,testset,_,_ = prepareData(batch_size=10)
    
    #net = Net() 
    net = ClassifierNet(input=28*28, output=10, hidden=20)
    #print(net)
    
    optimizer = optimizerTorch(net.parameters(), lr=1e-4)
    EPOCHS = 10
        
    #lr = float(optimizer.param_groups[0]["lr"])
    #print('lr=',lr)
    
    # x = torch.rand(28,28)
    # x = x.view(-1, 28*28)
    # outPut = net(x)
    # print(outPut)
    
    for epoch in range(EPOCHS):
        t = time.time()
        for data in trainset:
            X, y = data
            #print('X.shape=',X.shape)
            #print('y.shape=',y.shape)
            net.zero_grad()
            output = net(X.view(-1,28*28))
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            
        log= f'epoch[{epoch+1}/{EPOCHS}] loss={round(float(loss),4)},run in {round(time.time()-t,4)}s'
        print(log)
        writeLog(log + '\n')
            
    accuracy(net,trainset)
    saveModel(net,optimizer,epoch,loss,r'./res/')
           
if __name__ == '__main__':
    main()