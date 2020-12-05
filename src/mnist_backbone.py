#python3 Steven 12/05/20,Auckland,NZ
#pytorch backbone models
import torch
from commonTorch import ClassifierCNN_NetBB
from summaryModel import summaryNet
from backbones import*

def main():  
    nClass = 10 
    net = ClassifierCNN_NetBB(nClass, backbone=alexnet)
    summaryNet(net, (3,512,512))
    
    #net = ClassifierCNN_NetBB(nClass, backbone=vgg16)
    #summaryNet(net, (3,640,480))
    
    # net = ClassifierCNN_NetBB(nClass, backbone=resnet18)
    # summaryNet(net, (3,640,480))
    
    #net = ClassifierCNN_NetBB(nClass, backbone=squeezenet)
    #summaryNet(net, (3,640,480))
    
    ##net = ClassifierCNN_NetBB(nClass, backbone=densenet)
    ##summaryNet(net, (3, 512, 512))
    
    ##net = ClassifierCNN_NetBB(nClass, backbone=inception)
    ##summaryNet(net, (3,640,480))
    
    ##net = ClassifierCNN_NetBB(nClass, backbone=googlenet)
    ##summaryNet(net, (3,640,480))
    
    #net = ClassifierCNN_NetBB(nClass, backbone=shufflenet)
    #summaryNet(net, (3,640,480))
    
    #net = ClassifierCNN_NetBB(nClass, backbone=mobilenet)
    #summaryNet(net, (3,640,480))
    
    #net = ClassifierCNN_NetBB(nClass, backbone=resnext50_32x4d)
    #summaryNet(net, (3,640,480))
    
    #net = ClassifierCNN_NetBB(nClass, backbone=wide_resnet50_2)
    #summaryNet(net, (3,640,480))
    
    #net = ClassifierCNN_NetBB(nClass, backbone=mnasnet)
    #summaryNet(net, (3,640,480))
    return
    
if __name__ == '__main__':
    main()
    