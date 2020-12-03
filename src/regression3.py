#python3 Steven 11/21/20,Auckland,NZ
#pytorch: Multi-features regression model

import torch
import time
import torch.nn as nn
from commonTorch import RegressionNet,optimizerTorch,lossFunction
from commonTorch import preparDataSetMul,RegressionNet2

def predict(net, a):
    x = torch.tensor(a, dtype=torch.float) #torch.tensor([4],dtype=torch.float)
    pred = net(x)
    print(f'pred[{x}], result={float(pred)}')
    
def main():
    X,y = preparDataSetMul(N=2000, gamma=0.00001)
    
    net = RegressionNet2(input=2,hidden=20,hiddenlayers=2) #RegressionNet(hidden=10) 
    print(net)
   
    optimizer = optimizerTorch(net.parameters(), lr=1e-3)
    lossFuc = lossFunction()#mean suqare error
    lambda1 = lambda epoch: 0.8 ** (epoch//100)  #decay 0.5 every times
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    EPOCHS = 2800
    for epoch in range(EPOCHS):
        t = time.time()
        
        net.zero_grad()
        pred = net(X)
        #print(pred)
        loss = lossFuc(pred, y)
        #print('loss=', float(loss))
        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        if epoch % 100==0:
            lr = optimizer.param_groups[0]["lr"]
            log= f'epoch[{epoch+1}/{EPOCHS}] loss={round(float(loss),4)},lr={float(lr)},run in {round(time.time()-t,4)}s'
            print(log)

    predict(net, a=[0.1, 0.2])
    predict(net, a=[0.3, 0.1])
           
if __name__ == '__main__':
    main()