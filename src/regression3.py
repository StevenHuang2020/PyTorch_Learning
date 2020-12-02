#python3 Steven 11/21/20,Auckland,NZ
#pytorch: Multi-features regression model

import torch
import time

from commonTorch import RegressionNet,optimizerTorch,lossFunction,preparDataSet
    
def predict(net):
    x = torch.tensor([3.0]) #torch.tensor([4],dtype=torch.float)
    pred = net(x)
    print(f'pred[{float(x)}], result={float(pred)}')
    
def main():
    X,y = preparDataSet(N=200)
    net = RegressionNet() #RegressionNet(hidden=10) 
    #print(net)

    optimizer = optimizerTorch(net.parameters(), lr=1e-2)
    lossFuc = lossFunction()#mean suqare error
    
    EPOCHS = 800
    for epoch in range(EPOCHS):
        t = time.time()
        
        net.zero_grad()
        pred = net(X)
        loss = lossFuc(pred, y)
        #print('loss=',type(loss),loss)
        loss.backward()
        optimizer.step()
        
        if epoch % 50==0:
            log= f'epoch[{epoch+1}/{EPOCHS}] loss={round(float(loss),4)},run in {round(time.time()-t,4)}s'
            print(log)

    predict(net)
           
if __name__ == '__main__':
    main()