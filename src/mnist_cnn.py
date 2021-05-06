#python3 Steven 12/03/20,Auckland,NZ
#pytorch study
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from plotConfusionMatrix import plot_confusion_matrix

from commonTorch import ClassifierNet, ClassifierCNN_Net
from commonTorch import ClassifierCNN_Net2, ClassifierCNN_Net3
from commonTorch import optimizerTorch,optimizerDesc
from mnist_fc import prepareData,saveModel,writeLog,load_model
from plotLoss import plotLossAndAcc,plotFromLog

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
    return correct/total

def accuracy2(net, train):
    train.data = train.data.view(-1, 1, 28, 28)
    train.data = train.data.type(torch.FloatTensor)
    print('train.data.shape=', train.data.shape, train.data.dtype)
    print('train.targets.shape=', train.targets.shape, train.targets.dtype)
    preds = net(train.data.view(-1, 1, 28, 28))
    
    print('preds.shape=', preds.shape)
    preds_correct = get_num_correct(preds, train.targets)
    print('total correct:', preds_correct)
    acc = preds_correct / len(preds)
    #print('accuracy:', acc)
    return acc,preds
    
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
         
def get_all_preds(net, dataset):
    all_preds = torch.tensor([])
    with torch.no_grad():
        for batch in dataset:
            images, labels = batch
            preds = net(images)
            all_preds = torch.cat((all_preds, preds) ,dim=0)
        return all_preds
    
def evaluateModel(net,trainset,train):
    #acc = accuracy(net,trainset)
    acc,preds = accuracy2(net, train)
    print('Accuracy1:', round(acc,3))
    
    #confusion matrix
    # stacked = torch.stack((train.targets, preds.argmax(dim=1)), dim=1)
    # cmt = torch.zeros(10,10, dtype=torch.int64)
    # for p in stacked:
    #     tl, pl = p.tolist()
    #     cmt[tl, pl] = cmt[tl, pl] + 1
    # print(cmt)
    
    cmt = confusion_matrix(train.targets, preds.argmax(dim=1))
    plot_confusion_matrix(cmt, train.classes)
  
def main():
    trainset,testset,train,test = prepareData(batch_size=20000)
    
    curEpoch,curLoss = 0,0
    weightsDir = r'./res/weights/'
    #net = ClassifierNet(input=28*28, output=10, hidden=20) #Fc
    #net = ClassifierCNN_Net(10) #cnn
    #net = ClassifierCNN_Net2(10) #Sequential cnn
    net = ClassifierCNN_Net3(10)
    optimizer = optimizerTorch(net.parameters(), lr=1e-3)
    lossFuc = nn.CrossEntropyLoss() #nn.NLLLoss
    
    if 1:#continue training    
        net,optimizer,curEpoch,curLoss = load_model(net, optimizer, weightsDir)

    print(net)    
    #return 

    print('training start...')
    losse_list = []
    acc_list=[]
    EPOCHS = 30
    #optimizerDesc(optimizer)
    for epoch in range(EPOCHS):
        t = time.time()
        for data in trainset:
            X, y = data
            #print('X.shape=',X.shape, X.dtype)
            #print('y.shape=',y.shape, y.dtype)
            net.zero_grad()
            output = net(X)
            loss = lossFuc(output, y)
            loss.backward()
            optimizer.step()
            
        epoch = curEpoch + epoch
        #acc,_ = accuracy2(net,train)
        acc = accuracy(net,trainset)
        acc_list.append(round(acc,4))
        losse_list.append(float(loss))   
        
        log = f'epoch[{epoch+1-curEpoch}/{EPOCHS}][total:{epoch+1}], loss={round(float(loss),4)}, accuracy={round(acc,4)}, run in {round(time.time()-t,4)}s'
        print(log)
 
            
    saveModel(net, optimizer, epoch, loss, weightsDir)
    #plotLossAndAcc(losse_list, acc_list)
    plotFromLog(r'./log/log.txt')
    evaluateModel(net, trainset, train)
    
if __name__ == '__main__':
    main()