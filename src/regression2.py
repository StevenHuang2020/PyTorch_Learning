#python3 Steven 12/01/20,Auckland,NZ
#pytorch: Save Regression to gif

import numpy as np
import matplotlib.pyplot as plt
import torch
import os,time
import imageio
from commonTorch import RegressionNet,optimizerTorch,lossFunction,preparDataSet

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
                   
def main():
    #testLayers()
    X,y = preparDataSet(gamma=0.1)
    net = RegressionNet(hidden=50)
            
    ymin = float(torch.min(y))+1 #
    
    optimizer = optimizerTorch(net.parameters(), lr = 1e-2)
    lossFuc = lossFunction()
    
    my_images = []
    fig, ax = plt.subplots()
    
    EPOCHS = 400
    for epoch in range(EPOCHS):
        t = time.time()
      
        net.zero_grad()
        
        pred = net(X)
        #print(X.shape, pred.shape, y.shape)
        loss = lossFuc(pred, y)
        loss.backward()
        optimizer.step()
        
        # plot and show learning process
        plt.cla()
        ax.set_title('Regression Analysis', fontsize=12)
        ax.set_xlabel('Independent variable', fontsize=10)
        ax.set_ylabel('Dependent variable', fontsize=10)
        #ax.set_xlim(-1.05, 1.5)
        #ax.set_ylim(-0.25, 1.25)
        ax.scatter(X.data.numpy(), y.data.numpy(), color = "orange")
        ax.plot(X.data.numpy(), pred.data.numpy(), 'g-', lw=3)
        ax.text(2.2, ymin-1, 'Epoch = %d' % epoch, fontdict={'size': 10, 'color':  'red'})
        ax.text(2.2, ymin, 'Loss = %.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})

        # Used to return the plot as an image array 
        # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if EPOCHS<200 or (EPOCHS<500 and epoch % 2==0) or (epoch % 4==0) :
            my_images.append(image)

        if epoch % 50==0:
            log= f'epoch[{epoch+1}/{EPOCHS}] loss={round(float(loss),4)},run in {round(time.time()-t,4)}s'
            print(log)
        #plt.show()
        #break
    
    # save images as a gif    
    imageio.mimsave('./res/curve.gif', my_images, fps=20)   
           
if __name__ == '__main__':
    main()