#python3 Steven 12/01/20,Auckland,NZ
#pytorch: optimizer traning convergence speed

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import os,time
import imageio
from torchsummary import summary #pip install torchsummary

from commonTorch import RegressionNet,optimizerTorch,optimizerDesc
from commonTorch import lossFunction,preparDataSet
from common import getCsvDataset              
from plotLoss import plotLossDict

def plotGif(losses,label):
    my_images = []
    fig, ax = plt.subplots()
    
    for i in range(1, len(losses)):
        if i%50!=0:
            continue
            
        # plot and show learning process
        plt.cla()
        ax.set_title('Optimizer Loss speed', fontsize=12)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        #ax.set_xlim(-1.05, 1.5)
        #ax.set_ylim(-0.25, 1.25)
        #ax.scatter(X.data.numpy(), y.data.numpy(), color = "orange")
        #ax.plot(X.data.numpy(), pred.data.numpy(), 'g-', lw=3)
        ax.plot(range(i), losses[:i], 'g-', lw=3, label=label)
        #ax.text(0.75, 0.16, 'Epoch = %d' % epoch, transform=ax.transAxes, fontdict={'size': 10, 'color':  'red'})
        #ax.text(0.75, 0.12, 'Loss = %.4f' % loss.data.numpy(), transform=ax.transAxes, fontdict={'size': 10, 'color':  'red'})
        ax.legend()
        
        # Used to return the plot as an image array 
        # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
        my_images.append(image)
    
    # save images as a gif    
    imageio.mimsave('./res/curve.gif', my_images, fps=20)   
        
def plotGifDict(lossesDict,valueLen):
    my_images = []
    fig, ax = plt.subplots()
    
    for i in range(1, valueLen):
        if i%(valueLen//10) != 0: #every 50 times draw
            continue
            
        # plot and show learning process
        plt.cla()
        ax.set_title('Optimizer Loss speed', fontsize=12)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_xlim(0, valueLen)
        #ax.set_ylim(-0.25, 1.25)
        #ax.scatter(X.data.numpy(), y.data.numpy(), color = "orange")
        #ax.plot(X.data.numpy(), pred.data.numpy(), 'g-', lw=3)
        for key,value in lossesDict.items():
            ax.plot(range(i), value[:i], '.-', lw=0.5, label=key) #'g-', 
          
        ax.vlines(i, 0, 37000, linestyles='dashdot', colors='g')  
        #ax.text(0.75, 0.16, 'Epoch = %d' % epoch, transform=ax.transAxes, fontdict={'size': 10, 'color':  'red'})
        #ax.text(0.75, 0.12, 'Loss = %.4f' % loss.data.numpy(), transform=ax.transAxes, fontdict={'size': 10, 'color':  'red'})
        ax.legend()
        
        # Used to return the plot as an image array 
        # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
        my_images.append(image)
    
    # save images as a gif    
    imageio.mimsave('./res/curve.gif', my_images, fps=20)   

def trainNet(net,X,y,optimizer, EPOCHS = 400):
    lossFuc = lossFunction()
    #optimizerDesc(optimizer)
    losses=[]
    for epoch in range(EPOCHS):
        t = time.time()
      
        net.zero_grad()
        pred = net(X)
        #print(X.shape, pred.shape, y.shape)
        pred = pred.squeeze()
        loss = lossFuc(pred, y)
        loss.backward()
        optimizer.step()

        if epoch % (EPOCHS//10) == 0:
            log= f'epoch[{epoch+1}/{EPOCHS}] loss={round(float(loss),4)},run in {round(time.time()-t,4)}s'
            print(log)
        
        losses.append(float(loss))
    return losses            

def main():
    #X,y = preparDataSet(gamma=0.01) #random data
    #file = r'./db/fucDatasetReg_1F_NoLinear_100.csv' #fucDatasetReg_1F_100
    file = r'./db/fucDatasetReg_3F_1000.csv'
    X,_,y,_ = getCsvDataset(file)
    #print(X.shape)
    features = X.shape[1] 

    # net = RegressionNet(hidden=50)
    # optimizer = optimizerTorch(net.parameters(), lr = 1e-2)
    # losses = trainNet(net,X,y,optimizer)
    # print(len(losses),losses)
    # plotGif(losses, 'Adamax')
    
    epoches = 1000
    lr = 1e-3
    optimizers = []
    
    net = RegressionNet(input=features)
    #summary(net, (1,features))
    optimizers.append((net, optim.RMSprop(net.parameters(), lr=lr), 'RMSprop'))   
    net = RegressionNet(input=features)
    optimizers.append((net, optim.Adamax(net.parameters(), lr=lr), 'Adamax'))
    net = RegressionNet(input=features)
    optimizers.append((net, optim.Adam(net.parameters(), lr=lr), 'Adam'))
    net = RegressionNet(input=features)
    optimizers.append((net, optim.SGD(net.parameters(), lr=lr), 'SGD'))
    net = RegressionNet(input=features)
    optimizers.append((net, optim.Adadelta(net.parameters(), lr=lr), 'Adadelta'))
    net = RegressionNet(input=features)
    optimizers.append((net, optim.Adagrad(net.parameters(), lr=lr), 'Adagrad'))
    net = RegressionNet(input=features)
    optimizers.append((net, optim.AdamW(net.parameters(), lr=lr), 'AdamW'))    
    net = RegressionNet(input=features)
    optimizers.append((net, optim.ASGD(net.parameters(), lr=lr), 'ASGD'))
    net = RegressionNet(input=features)
    optimizers.append((net, optim.Rprop(net.parameters(), lr=lr), 'Rprop'))
 
    ##optimizers.append((net, optim.SparseAdam(net.parameters(), lr=lr), 'SparseAdam'))
    ##optimizer = optim.LBFGS(net.parameters(), lr=lr)

    lossesDict={}
    for net, opt, name in optimizers:
        losses = trainNet(net,X,y,opt,EPOCHS=epoches)
        lossesDict[name]=losses
        
    plotGifDict(lossesDict,epoches)
    plotLossDict(lossesDict, 'Optimizer Loss speed')
    
if __name__ == '__main__':
    main()
    