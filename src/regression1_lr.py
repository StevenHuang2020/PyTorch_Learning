#python3 Steven 12/01/20,Auckland,NZ
#pytorch: Dynamically adjust learning rate

import torch
import time
import matplotlib.pyplot as plt
from commonTorch import RegressionNet,optimizerTorch,lossFunction,preparDataSet

def predict(net):
    x = torch.tensor([3.0]) #torch.tensor([4],dtype=torch.float)
    pred = net(x)
    print('pred result=',float(pred))

def main():
    #return testLr()
    
    X,y = preparDataSet(N=200)
    net = RegressionNet() #RegressionNet(hidden=10) 
    #print(net)

    optimizer = optimizerTorch(net.parameters(), lr=1e-1)
    lossFuc = lossFunction()#mean suqare error
    lambda1 = lambda epoch: 0.5 ** (epoch//200)  #decay 0.5 every 50 times
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    EPOCHS = 800
    for epoch in range(EPOCHS):
        t = time.time()
        
        net.zero_grad()
        pred = net(X)
        loss = lossFuc(pred, y)
        #print('loss=',type(loss),loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 50==0:
            log= f'epoch[{epoch+1}/{EPOCHS}] loss={round(float(loss),4)},run in {round(time.time()-t,4)}s'
            print(log)
            #print('epoch:',epoch, 'lr:',optimizer.param_groups[0]["lr"])
    
    predict(net)
            
def testLr():#adjust learning rate according to a scheduler
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=100)
    lambda1 = lambda epoch: 0.5 ** epoch
    #lambda1 = lambda epoch: epoch // 2
    #lambda1 = lambda epoch: 0.5 ** (epoch//10)
    
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) #lr = lr_start * Lambda(epoch)
    #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda1) #lr_i+1 = lr_i * Lambda(epoch)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,8,9], gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=0)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=3,mode="triangular")#mode=triangular2
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="exp_range",gamma=0.85)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10,anneal_strategy='linear')
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1)
    
    N = 10
    lrs = []
    for i in range(N):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        #print("Factor = ", round(0.65 ** i,3)," , Learning Rate = ",round(optimizer.param_groups[0]["lr"],3))
        scheduler.step()

    print(lrs)
    '''
    LambdaLR: [100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.390625, 0.1953125]
    MultiplicativeLR: [100, 50.0, 12.5, 1.5625, 0.09765625, 0.0030517578125, 4.76837158203125e-05, 3.725290298461914e-07, 1.4551915228366852e-09, 2.8421709430404007e-12]
    StepLR: [100, 100, 50.0, 50.0, 25.0, 25.0, 12.5, 12.5, 6.25, 6.25]
    MultiStepLR: [100, 100, 100, 100, 100, 50.0, 50.0, 50.0, 25.0, 12.5]
    ExponentialLR: [100, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.390625, 0.1953125]
    CosineAnnealingLR: [100, 50.0, 0.0, 49.99999999999999, 100.00000000000001, 50.00000000000002, 0.0, 49.99999999999999, 100.00000000000003, 50.000000000000036]
    CyclicLR: [0.001, 0.034000000000000016, 0.06699999999999999, 0.1, 0.06700000000000003, 0.033999999999999975, 0.001, 0.03400000000000006, 0.06699999999999995, 0.1]
    '''
    
    plt.plot(range(N),lrs)
    plt.show()
    
if __name__ == '__main__':
    main()