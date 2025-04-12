import numpy as np
import torch 
import torch.nn as nn   
from torch.autograd import Variable
from torch.optim import Adam    # an optimizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import time
from convNet import ConvNet
from hybridNet import ConvNet as hybridNet
from thinNet import thinNet
from lowKernelNet import lkNet

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

#normalizes dataset so every EEG channel has a mean of 0 and standard deviation of 1 across all examples.
def setNorm(dataset):
    for i in range(dataset.shape[1]):
        channelMeans = dataset[:,i,:].mean()
        channelStdDev = dataset[:,i,:].std()
        dataset[:,i,:] = (dataset[:,i,:] - channelMeans)/channelStdDev
    return dataset


def tester(net, netArch, data, targets, batch_size, checkpointFile="None"):
    data = setNorm(data)
    X_test, Y_test = data, targets
    

    if os.path.exists(checkpointFile):
        checkpoint = torch.load(checkpointFile)
        net.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded checkpoint from file: ", checkpointFile)
    else:
        print("No checkpoint found at: ", checkpointFile)
        return
    
    net.eval()

    if netArch == hybridNet:
        curr_testAcc = testModelHybrid(net, X_test, Y_test, batch_size)
    else:
        curr_testAcc = testModel(net, X_test, Y_test, batch_size)

    return curr_testAcc
    


""" 
Model testing function, prints accuracy of classifier on input test data and labels
    net: trained model to test
    X_test, Y_test: data and labels, respectively
"""
def testModel(net, X_test, Y_test, batch_size):
    with torch.no_grad():
        test_loss = 0.0
        test_total = 0
        for i in range(int(len(X_test)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
            inputs = X_test[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_test[s:e]
            outputs = net(inputs.cuda(0))
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.to(device='cpu')
            total = len(labels)
            correct = (predicted == labels).sum().item()
            test_loss += correct
            test_total += total
            del inputs
            del labels
            del _
    return (100 * test_loss / test_total)

def testModelHybrid(net, X_test, Y_test, batch_size):
    with torch.no_grad():
        test_loss = 0.0
        test_total = 0
        for i in range(int(len(X_test)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
            inputs = X_test[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_test[s:e]
            outputs, _ = net(inputs.cuda(0))
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.to(device='cpu')
            total = len(labels)
            correct = (predicted == labels).sum().item()
            test_loss += correct
            test_total += total
            del inputs
            del labels
            del _
    return (100 * test_loss / test_total)
        

def mainLoop(netArch, data, targets,  num_classes, batch_size=50, checkpointFile="None"):
    # Defining the loss function and optimizer
    net = netArch(num_classes=num_classes).cuda(0)
    teAcc = tester(net = net, 
                netArch = netArch,
                data = data, 
                targets = targets, 
                batch_size = batch_size, 
                checkpointFile = checkpointFile)

    return teAcc

def main(netArch,name, checkpointName):
    fileDataPath = 'scratch/data/'
    fileTargetPath = 'scratch/target/'
    checkpointFile = "checkpoint/"+checkpointName

    if os.path.exists(fileDataPath+name+"_data.pt") and os.path.exists(fileTargetPath+name+"_target.pt"):
        print("\n\n\n")
        print("==========================")
        print("Analyzing dataset: ", fileDataPath+name," and ", fileTargetPath+name)
        data = torch.load(fileDataPath+name+"_data.pt")   # location of the dataset
        targets = torch.load(fileTargetPath+name+"_target.pt")
        print(data.shape[0])
        teAcc = mainLoop(netArch=netArch,
                        data = data,
                        targets = targets, 
                        num_classes=3, 
                        batch_size=281, 
                        checkpointFile=checkpointFile)
        
        print("average test acc for dataset is ", teAcc)
        print("==========================")
        print("\n\n\n")

        return teAcc

# main(netArch=hybridNet,
#     name = "test_250", 
#     checkpointName = "checkpointHybridNet.pth"
#     )

