#python3 Steven 12/03/20,Auckland,NZ
#pytorch study
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from plotConfusionMatrix import plot_confusion_matrix

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
    print('Accuracy1: ', round(correct/total,3))

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
    accuracy(net,trainset)
    preds = get_all_preds(net, trainset)
    print('preds.shape=', preds.shape)
    preds_correct = get_num_correct(preds, train.targets)
    print('total correct:', preds_correct)
    print('accuracy:', preds_correct / len(preds))
    
    #confusion matrix
    # stacked = torch.stack((train.targets, preds.argmax(dim=1)), dim=1)
    # cmt = torch.zeros(10,10, dtype=torch.int64)
    # for p in stacked:
    #     tl, pl = p.tolist()
    #     cmt[tl, pl] = cmt[tl, pl] + 1
    # print(cmt)
    
    cmt = confusion_matrix(train.targets, preds.argmax(dim=1))
    plot_confusion_matrix(cmt, train.classes)
  
def plotLoss(loss,name='Loss'):
    plt.title(name)
    plt.plot(loss)
    plt.show()  
    
def main():
    trainset,testset,train,test = prepareData(batch_size=10)
    #net = ClassifierNet(input=28*28, output=10, hidden=20) #Fc
    #net = ClassifierCNN_Net(10) #cnn
    #net = ClassifierCNN_Net2(10) #Sequential cnn
    net = ClassifierCNN_Net3(10)
    print(net)
    
    optimizer = optimizerTorch(net.parameters(), lr=1e-6)
    EPOCHS = 10
    #optimizerDesc(optimizer)

    print('training start...')
    losses = []
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
            losses.append(float(loss))
            
        log= f'epoch[{epoch+1}/{EPOCHS}] loss={round(float(loss),4)},run in {round(time.time()-t,4)}s'
        print(log)
        #writeLog(log + '\n')
            
    #saveModel(net,optimizer,epoch,loss,r'./res/')
    plotLoss(losses)
    evaluateModel(net, trainset, train)
    
   
if __name__ == '__main__':
    main()