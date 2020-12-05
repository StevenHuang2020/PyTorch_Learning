#python3 Steven 12/05/20,Auckland,NZ
#pytorch: Classification

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,time
from torchsummary import summary #pip install torchsummary
from common import getCsvDataset,getExcelDataset              
from commonTorch import ClassifierNet,optimizerTorch,lossFunction

def descpritDataset(X, y):
    total = 0
    counter_dict={0:0,1:0,2:0}
    for yi in y:
        counter_dict[int(yi)] += 1
        total += 1
            
    print(counter_dict,'total=',total)
    for i in counter_dict:
        print(f"{i}: {round(counter_dict[i]*100/total,4)}%")

def accuracy(net,X,y):
    correct = 0
    total = 0
    output = net(X)
    for idx, i in enumerate(output):
        correct = correct + int(torch.argmax(i) == y[idx])
        total += 1
    print(f'correct:{correct},total:{total}, Accuracy:{round(correct/total,3)}')
    
def trainNet(net,X,y,optimizer, EPOCHS = 400, lossFuc = lossFunction()):
    losses=[]
    for epoch in range(EPOCHS):
        t = time.time()
      
        net.zero_grad()
        pred = net(X)
        #print('X.shape, pred.shape, y.shape=', X.shape, pred.shape, y.shape)
        #print('pred.dtype, y,dtype=', pred.dtype, y.dtype)
        loss = lossFuc(pred, y)
        loss.backward()
        optimizer.step()

        if epoch % (EPOCHS//10) == 0:
            log= f'epoch[{epoch+1}/{EPOCHS}] loss={round(float(loss),4)},run in {round(time.time()-t,4)}s'
            print(log)
        
        losses.append(float(loss))
    return losses            

def main():
    #file = r'./db/fucDatasetClf_2F_MClass_1000.csv'
    #X,_,y,_ = getCsvDataset(file)
    file = r'./db/Iris.xlsx'
    X,_,y,_ = getExcelDataset(file)
    
    y = y.type(torch.long)
    
    descpritDataset(X,y)
    
    #print(X.shape)
    features = X.shape[1] 
    labels= np.unique(y)
    print('features=',features,'labels=', labels)
    
    epoches = 800
    lr = 1e-2
    
    lossFuc = nn.CrossEntropyLoss()
    #lossFuc = nn.NLLLoss()
    
    net = ClassifierNet(input=features, output=len(labels), hidden=20)
    summary(net,(1,features))
    optimizer = optimizerTorch(net.parameters(), lr = lr)
    trainNet(net, X, y, optimizer, EPOCHS=epoches, lossFuc=lossFuc)
    
    accuracy(net,X,y)

if __name__ == '__main__':
    main()
    