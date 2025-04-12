import numpy as np
import torch 
import torch.nn as nn   
from torch.autograd import Variable
from torch.optim import Adam    # an optimizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from convNet import ConvNet
from hybridNet import ConvNet as hybridNet
from thinNet import thinNet
from lowKernelNet import lkNet
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import time
import csv

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


#shuffles data
def shuffleData(inData, labels):
    reshuffleIndex = np.random.permutation(len(inData))
    return torch.tensor(inData.numpy()[reshuffleIndex]), torch.tensor(labels.numpy()[reshuffleIndex]), reshuffleIndex


#normalizes dataset so every EEG channel has a mean of 0 and standard deviation of 1 across all examples.
def setNorm(dataset):
    for i in range(dataset.shape[1]):
        channelMeans = dataset[:,i,:].mean()
        channelStdDev = dataset[:,i,:].std()
        dataset[:,i,:] = (dataset[:,i,:] - channelMeans)/channelStdDev
    return dataset

#Adds Gaussian noise with input standard deviation to dataset.
def addNoise(dataset, stdDev):
    randomAdd = np.random.normal(0, stdDev, dataset.shape)
    noisedData = torch.tensor(dataset.numpy() + randomAdd).type(torch.FloatTensor)
    return noisedData


def createTrainTestSplit(data, targets):
    X_train = data[:47208]   
    Y_train = targets[:47208]
    print("X_train shape: ", X_train.shape)
    print("Y_train shape: ", Y_train.shape)
    return X_train, Y_train


"""
The main training function for our model. Randomizes dataset on each epoch and adds noise if noise param !=0. 
    net: the model to train and test
    optimizer: specified optimizer
    num_epoch: number of epoch to run during training 
    noise: proportion of Gaussian noise with input standard deviation to dataset
    returns lists of accuracy at each epoch on training and test sets.
"""
def trainModel(net, netArch, dataTrain, targetsTrain, dataValid, targetsValid, optimizer, loss_fn, num_epochs, noise, batch_size, printout = True, tensor_board = True, checkpointFile="None"):
    trainAcc, testAcc = [], []  

    dataTrain = setNorm(dataTrain)   # normalize the data    
    dataValid = setNorm(dataValid)   # normalize the data
    X_train, Y_train = dataTrain, targetsTrain 
    #X_train, Y_train = createTrainTestSplit(dataTrain, targetsTrain) 
    X_test, Y_test = dataValid, targetsValid 
    timeList = []
    for epoch in range(num_epochs):
        start_time = time.time()
        net.train()  # Set the model to training mode
        # Load checkpoint if it exists
        if os.path.exists(checkpointFile):
            checkpoint = torch.load(checkpointFile)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded checkpoint from file: ", checkpointFile)
        else:
            print("No checkpoint found at: ", checkpointFile)
            print("Starting from scratch")

        if printout:
            print("\n Epoch: ", epoch)
        X_epoch, Y_epoch, shufPerm = shuffleData(X_train, Y_train) 
        
        if noise != 0:
            X_epoch = addNoise(X_epoch, noise)
            X_epoch = setNorm(X_epoch)
        running_loss = 0.

        for i in range(int(len(X_epoch)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
            
            inputs = X_epoch[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_epoch[s:e]
            inputs, labels = Variable(inputs.cuda(0)), Variable(labels.type(torch.LongTensor).cuda(0))
            optimizer.zero_grad() # clear gradients       
            outputs = net(inputs)  # forward propagation
            loss= loss_fn(outputs, labels)   # calculate the loss
            loss.backward()   # Calculating gradients
            optimizer.step()  # Update parameters

            running_loss += float(loss.item())
        
        del loss 
        del labels
        del inputs 
        del outputs 

        net.eval()
        curr_trainAcc = testModel(net, X_train, Y_train, batch_size, netArch)
        curr_testAcc = testModel(net, X_test, Y_test, batch_size, netArch)
        

        if printout:
            print("Training Loss:", running_loss)
            print("Accuracy on the train set: {} %".format(curr_trainAcc))
            print("Accuracy on the valid set: {}%".format(curr_testAcc))
        
        trainAcc.append(curr_trainAcc)
        testAcc.append(curr_testAcc)

        end_time = time.time()
        print("Epoch time: ", end_time - start_time)
        timeList.append(end_time - start_time)


    #save checkpoint
    checkpoint = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpointFile)
    print("Saved checkpoint to file: ", checkpointFile)
    print("Average time per epoch: ", np.mean(timeList))
    return trainAcc, testAcc
    
def trainHybridModel(net, netArch, dataTrain, targetsTrain, dataValid, targetsValid, optimizer, loss_fn, num_epochs, noise, batch_size, printout = True, tensor_board = True, checkpointFile="None"):
    trainAcc, testAcc = [], []  
    AE_loss_fn = nn.MSELoss()
    dataTrain = setNorm(dataTrain)   # normalize the data    
    dataValid = setNorm(dataValid)   # normalize the data
    #X_train, Y_train = createTrainTestSplit(dataTrain, targetsTrain) 
    X_train, Y_train = dataTrain, targetsTrain 
    X_test, Y_test = dataValid, targetsValid 
    timeList = []

    for epoch in range(num_epochs):
        start_time = time.time()
        net.train()  # Set the model to training mode
        # Load checkpoint if it exists
        if os.path.exists(checkpointFile):
            checkpoint = torch.load(checkpointFile)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded checkpoint from file: ", checkpointFile)
        else:
            print("No checkpoint found at: ", checkpointFile)
            print("Starting from scratch")

        if printout:
            print("\n Epoch: ", epoch)
        X_epoch, Y_epoch, shufPerm = shuffleData(X_train, Y_train) 
        
        if noise != 0:
            X_epoch = addNoise(X_epoch, noise)
            X_epoch = setNorm(X_epoch)
        running_loss = 0.

        for i in range(int(len(X_epoch)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
            
            inputs = X_epoch[s:e].unsqueeze(1).type(torch.FloatTensor)
            inputsUN = X_epoch[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_epoch[s:e]
            inputs, labels = Variable(inputs.cuda(0)), Variable(labels.type(torch.LongTensor).cuda(0))
            inputsUN = Variable(inputsUN.cuda(0))
            optimizer.zero_grad() # clear gradients       
            CLout, AEout = net(inputs)  # forward propagation
            
            CLloss = loss_fn(CLout, labels)   # calculate the loss
            AEloss = AE_loss_fn(AEout, inputsUN)   # calculate the loss

            loss = 0.4 * AEloss + 0.6 * CLloss

            loss.backward()   # Calculating gradients
            optimizer.step()  # Update parameters

            running_loss += float(loss.item())
            del CLloss
            del AEloss
            del labels
            del inputs 
            del AEout
            del CLout
            del inputsUN
        
        

        net.eval()
        curr_trainAcc = testModel(net, X_train, Y_train, batch_size, netArch)
        curr_testAcc = testModel(net, X_test, Y_test, batch_size, netArch)
        

        if printout:
            print("Training Loss:", running_loss)
            print("Accuracy on the train set: {} %".format(curr_trainAcc))
            print("Accuracy on the test set: {}%".format(curr_testAcc))
        
        trainAcc.append(curr_trainAcc)
        testAcc.append(curr_testAcc)

        end_time = time.time()
        print("Epoch time: ", end_time - start_time)
        timeList.append(end_time - start_time)


    #save checkpoint
    checkpoint = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpointFile)
    print("Saved checkpoint to file: ", checkpointFile)
    print("Average time per epoch: ", np.mean(timeList))
    return trainAcc, testAcc
    


""" 
Model testing function, prints accuracy of classifier on input test data and labels
    net: trained model to test
    X_test, Y_test: data and labels, respectively
"""
def testModel(net, X_test, Y_test, batch_size, netArch = ConvNet):
    with torch.no_grad():
        test_loss = 0.0
        test_total = 0
        for i in range(int(len(X_test)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
            inputs = X_test[s:e].unsqueeze(1).type(torch.FloatTensor)
            labels = Y_test[s:e]
            if netArch == hybridNet:
                outputs, _ = net(inputs.cuda(0))
            else:
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


def mainLoop(netArch, dataTrain, targetsTrain, dataValid, targetsValid,  num_classes, learning_rate = 0.0003, weight_decay=0.003, batch_size=50, num_epochs = 10, noise = 0.0, dropout=0.0, checkpointFile="None"):
    dataTrain, targetsTrain, shufPerm = shuffleData(dataTrain, targetsTrain) # shuffle the data
    # Defining the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    net = netArch(num_classes=num_classes,dropout=dropout).cuda(0)
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)
    
    print("Model: ", netArch)
    if netArch == hybridNet:
        trAcc, teAcc = trainHybridModel(net, netArch, dataTrain, targetsTrain, dataValid, targetsValid, optimizer, loss_fn, num_epochs, noise, batch_size, checkpointFile = checkpointFile) 
    else:
        trAcc, teAcc = trainModel(net, netArch, dataTrain, targetsTrain, dataValid, targetsValid, optimizer, loss_fn, num_epochs, noise, batch_size, checkpointFile = checkpointFile) 
    
    return trAcc, teAcc


def main(netArch, nameTrain = "", nameValid = "" ,checkpointName = "None", learning_rate = 0.0003, weight_decay=0.003, batch_size=50, num_epochs = 10, noise = 0.0, dropout=0.0):
    fileDataPath = 'scratch/data/'
    fileTargetPath = 'scratch/target/'

    checkpointFile = "checkpoint/"+checkpointName

    print("\n\n\n")
    print("==========================")
    print("Analyzing dataset: ", fileDataPath+nameTrain," and ", fileTargetPath+nameTrain)
    print("Analyzing dataset: ", fileDataPath+nameValid," and ", fileTargetPath+nameValid)

    dataTrain = torch.load(fileDataPath+nameTrain+"_data.pt")   # location of the dataset
    targetsTrain= torch.load(fileTargetPath+nameTrain+"_target.pt")

    dataValid = torch.load(fileDataPath+nameValid+"_data.pt")   # location of the dataset
    targetsValid = torch.load(fileTargetPath+nameValid+"_target.pt")

    print(dataTrain.shape[0])
    trAcc, teAcc = mainLoop(netArch=netArch,
                            dataTrain = dataTrain,
                            targetsTrain = targetsTrain,
                            dataValid = dataValid,
                            targetsValid = targetsValid,  
                            num_classes=3, 
                            learning_rate = learning_rate, 
                            weight_decay= weight_decay, 
                            batch_size= batch_size, 
                            num_epochs = num_epochs, 
                            noise = noise, 
                            dropout= dropout, 
                            checkpointFile = checkpointFile
                            )

    print("train acc & test acc for dataset is ", trAcc, teAcc)
    print("average train acc & test acc for dataset is ", np.mean(trAcc), np.mean(teAcc))
    print("==========================")
    print("\n\n\n")
    return trAcc, teAcc

# main(netArch=hybridNet,
#     nameTrain ="train_250",
#     nameValid = "valid_250",
#     checkpointName = "checkpointHydribNet_100.pth",
#     learning_rate = 0.0003, 
#     weight_decay= 0.001, 
#     batch_size= 1, 
#     num_epochs = 5, 
#     noise = 0.1, 
#     dropout= 0.3, 
#     )