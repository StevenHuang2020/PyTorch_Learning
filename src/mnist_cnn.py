#python3 Steven 12/03/20,Auckland,NZ
#pytorch study
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from commonTorch import ClassifierNet, ClassifierCNN_Net
from commonTorch import ClassifierCNN_Net2, ClassifierCNN_Net3
from commonTorch import optimizerTorch,optimizerDesc
from mnist_fc import prepareData,saveModel,writeLog
   
def accuracy(net,dataset):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset:
            X,y = data
            output = net(X)
            for idx, i in enumerate(output):
                #if torch.argmax(i) == y[idx]:
                    #correct += 1
                correct = correct + int(torch.argmax(i) == y[idx])
                total += 1
    print('Accuracy: ', round(correct/total,3))
                
def main():
    trainset,testset = prepareData(batch_size=10)
    #net = ClassifierNet(input=28*28, output=10, hidden=20) #Fc
    #net = ClassifierCNN_Net(10) #cnn
    #net = ClassifierCNN_Net2(10) #Sequential cnn
    net = ClassifierCNN_Net3(10)
    print(net)
    
    optimizer = optimizerTorch(net.parameters(), lr=1e-5)
    EPOCHS = 5
    #optimizerDesc(optimizer)

    print('training start...')
    for epoch in range(EPOCHS):
        t = time.time()
        for data in trainset:
            X, y = data
            # print('X.shape=',X.shape)
            # print('y.shape=',y.shape)
            net.zero_grad()
            output = net(X)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            
        log= f'epoch[{epoch+1}/{EPOCHS}] loss={round(float(loss),4)},run in {round(time.time()-t,4)}s'
        print(log)
        #writeLog(log + '\n')
            
    accuracy(net,trainset)
    #saveModel(net,optimizer,epoch,loss,r'./res/')
           
if __name__ == '__main__':
    main()