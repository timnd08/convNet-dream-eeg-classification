"""
Dataloader program that creates a Pytorch compatible tensor from a raw tab-separated txt file 
"""
import csv
from numpy.core.numeric import full
import pandas as pd
import numpy as np
import torch
import sys
import os
csv.field_size_limit(sys.maxsize)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

def selectAndAddChannel(eventgrp, name):
    name = str(name)
    channel = eventgrp[eventgrp['channel'] == name]['data'].values
    channel_np = np.fromstring(channel[0], dtype=float, sep=",")[:250]
    return eventgrp, channel_np


def createTensorGroup(eventgrp, transform=True, exclude=[]):
    list_of_channels=["Fp1","F3","F7","Fp2","F4","F8","C3","C4","O2","Fz","Cz","Pz","Oz","EMG2"]    
    [Fp1np,F3np,F7np,Fp2np,F4np,F8np,C3np,C4np,O2np,Fznp,Cznp,Pznp,Oznp,EMG2np] =\
        [None, None, None, None,None, None, None, None, None, None, None, None, None, None]       
    NoneType = type(Fp1np)
    list_of_vars = [Fp1np,F3np,F7np,Fp2np,F4np,F8np,C3np,C4np,O2np,Fznp,Cznp,Pznp,Oznp,EMG2np]

    for i in range(len(list_of_channels)):
        eventgrp, list_of_vars[i] = selectAndAddChannel(eventgrp, list_of_channels[i])

    list_of_vars = [x for x in list_of_vars if type(x) != NoneType]  # remove None channels  #todo: the

    fullChannel = torch.tensor(np.vstack(list_of_vars))
    target = int(eventgrp.iloc[0]['code'])
    corrLen = True

    if list(fullChannel.size())[1] < 200:       # remove all instances shorter than 250 events
        print("Short data length")
        corrLen = False
    
    if len(list_of_vars) < (14-len(exclude)):
        print("Fewer channels than expected - ", len(list_of_vars))
        corrLen = False

    return fullChannel, target, corrLen


def dataLoader(filename, exclude, outpathData, outpathTargets):
    lines = []
    lineMax = 0
    
    with open(filename) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):
            if lineMax == 2000000:  # change this line to select how many examples the dataset to contain
                print("maxLines reached", lineMax)
                break
            lines.append(line)
            lineMax += 1

    dF = pd.DataFrame.from_records(lines, columns=['id', 'event', 'device', 'channel', 'code', 'size', 'data'])
    cleanDf = dF.drop(['id', 'device'], axis=1)
    cleanDf['size'] = pd.to_numeric(cleanDf['size'])
    cleanDf['event'] = pd.to_numeric(cleanDf['event'])
    minSize = cleanDf['size'].min()

    events = cleanDf['event'].to_numpy()
    numEvents = len(events)

    eventGroups = []
    targets = []
    eventTensors = []

    #print("Grouping event channels")
    for event in events:
        eventGroups.append(cleanDf.loc[cleanDf['event'] == event])


    for eventGroup in eventGroups:
        tensor, target, corrLen = createTensorGroup(eventGroup, transform=True, exclude=[])
        if corrLen:
            eventTensors.append(tensor)
            targets.append(target)

    fullDataTensor = torch.stack(eventTensors)
    targetTensor = torch.tensor(targets)
    torch.save(fullDataTensor, outpathData)
    torch.save(targetTensor, outpathTargets)


filePath = "dataset/"
outpathData = "scratch/data/" # change this line if the location of the file has changed
outpathTargets = "scratch/target/" # change this line if the location of the file has changed]

names = ["test_250","train_250","valid_250"]

for name in names:
    print("Processing ", name)
    dataLoader(filename=filePath+name+".txt", 
                exclude=[], 
                outpathData= outpathData+name+"_data.pt", 
                outpathTargets=outpathTargets+name+"_target.pt")