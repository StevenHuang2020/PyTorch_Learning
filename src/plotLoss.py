#Steven plot loss & acc from file
import argparse 
import sys
import matplotlib.pyplot as plt

#----------------------------------------------
#usgae: python plotloss.py .\log\log.txt
#----------------------------------------------

def getLoss(log_file):
    numbers = {'1','2','3','4','5','6','7','8','9'}
    iters = []
    loss_list = []
    accuracy=[]
    val_loss = []
    val_accuracy = []
    
    with open(log_file, 'r') as f:
        lines  = [line.rstrip("\n") for line in f.readlines()]

        epoch = 0
        for line in lines:
            trainIterRes = line.split(' ')
            #print(trainIterRes)
            if trainIterRes[0].startswith('epoch') and trainIterRes[1].startswith('loss') \
                and trainIterRes[2].startswith('accuracy'):
                epoch = int(trainIterRes[0][trainIterRes[0].rfind(':')+1 : -2])#epoch+=1
                loss = float(trainIterRes[1][trainIterRes[1].rfind('=')+1 : -1])
                acc = float(trainIterRes[2][trainIterRes[2].rfind('=')+1 : -1])
                
                #print('epoch,loss,acc=',epoch,loss,acc)
                iters.append(epoch)
                loss_list.append(loss)
                accuracy.append(acc)
                
                #val_loss.append(float(trainIterRes[13]))
                #val_accuracy.append(float(trainIterRes[16]))

    return iters,loss_list,accuracy,val_loss,val_accuracy

def plotLoss(loss,name='Loss'):
    plt.title(name)
    plt.plot(loss)
    plt.show()  
    
def plotLossAndAcc(loss,acc,name='Loss & Accuracy'):
    plt.title(name)
    plt.plot(loss, label='Loss')
    plt.plot(acc, label='Accuracy')
    plt.hlines(y=1,xmin=0, xmax=len(loss), colors='g', linestyles='dashed')
    plt.legend()
    plt.tight_layout()
    #plt.ylabel('Epoch')
    plt.xlabel('Epoch')
    plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=None, hspace=None)
    plt.savefig(r'./res/loss.png')
    plt.show()
    
def plotLossDict(lossesDict, name='Loss'):
    plt.cla()
    plt.title(name)
    for key,value in lossesDict.items():
        #print('dict:', key)
        plt.plot(value,label=key)
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.ylim(0,1300)
    #plt.xlim(0,111)
    plt.savefig(r'./res/lossDict.png')
    plt.show()
    
def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--list', nargs='+', help='path to log file', required=True)
    parser.add_argument('-s', '--start', help = 'startIter')
    parser.add_argument('-t', '--stop', help = 'stopIter')
    return parser.parse_args()

def plotFromLog(logFile):
    iters,loss,acc,val_loss,val_acc = getLoss(logFile)

    #print('loss=',loss)
    #print('val_loss=',val_loss)
    #print('acc=',acc)
    #print('val_acc=',val_acc)
    
    #plotLoss(iters,loss,val_loss)
    #plotAcc(iters,acc,val_acc)
    plotLossAndAcc(loss,acc)
    
def main():
    args = argCmdParse()
    
    # startIter = 0
    # stopIter = None
    # if args.start:
    #     startIter = int(args.start)
    # if args.stop:
    #     stopIter = int(args.stop)
        
    print(args.list)
    file = args.list[0]
    plotFromLog(file)
    
if __name__ == "__main__":
    main()
    