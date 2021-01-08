import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from starter.Tree import Tree
from starter.Node import Node


def getPossibleFeatureValues(feature):
    possibleValues=[]
    for x in feature:
        if not(x in possibleValues):
            possibleValues.append(x)
    return possibleValues


def completeTree(node,dataset):
    dataSetEntropy=getFeatureEntropy(dataset['output'].values.tolist())
    if(dataSetEntropy==0):
        return
    gainValuesPerFeature = []
    for element in columnHeader:
        if element=='output':
            gainValuesPerFeature.append(0)
            continue
        feature = dataset[element].values.tolist()
        gainValuesPerFeature.append(calculateGainForAttribuite(feature,dataSetEntropy,dataset['output'].values.tolist()))
    maxGain = max(gainValuesPerFeature)
    maxIndex = gainValuesPerFeature.index(maxGain)
    node.name=columnHeader[maxIndex]
    possbileValues=getPossibleFeatureValues(dataset[columnHeader[maxIndex]].values.tolist())

    for index,value in enumerate(possbileValues):
        node.children.append(Node())
        completeTree(node.children[index],dataset[dataSet[node.name]==value])


def calculateEntropy(numberOfYes, numberOfNo):
    if(numberOfNo==0 and numberOfYes==0):
        return 0
    sum = numberOfYes + numberOfNo
    y=0
    k=0
    x=-1*numberOfYes / sum
    if(x!=0):
        y=math.log(float(numberOfYes) / float(sum),2)
    z=numberOfNo / sum
    if(z!=0):
        k=math.log(float(numberOfNo) / float(sum),2)
    return (x* y -1* z*k)


def getFeatureEntropy(outputs):
    democrates = 0
    republicans = 0
    for i in range(len(outputs)):
        if (outputs[i] == 'democrat'):
            democrates += 1
        else:
            republicans += 1
    print("REPUBLICANS: ",republicans)
    print("DEMOCRATES: ",democrates)
    return calculateEntropy(democrates, republicans)


def getValueEntropy(valueVoters,outputs):
    democrates = 0
    republicans = 0
    for index in valueVoters:
        if (outputs[index] == 'democrat'):
            democrates += 1
        else:
            republicans += 1
    return calculateEntropy(democrates, republicans)


def getDemocratesnumber(valueArray,outputs):
    democrates = 0
    x=outputs
    for index in valueArray:
        if (outputs[index] == 'democrat'):
            democrates += 1
    return democrates


def calculateGainForAttribuite(feature,featureEntropy,outputs):
    yesVoters = []
    noVoters = []
    for index in range(len(feature)):
        if (feature[index] == 'y'):
            yesVoters.append(index)
        elif (feature[index] == 'n'):
            noVoters.append(index)
    yesEntropy = 0
    noEntropy = 0
    if(len(yesVoters)!=0):
        yesEntropy = getValueEntropy(yesVoters,outputs)
    if(len(noVoters)!=0):
        noEntropy = getValueEntropy(noVoters,outputs)

    firstValueEntropy=(len(yesVoters) / len(feature) * yesEntropy)
    secondValueEntropy=(len(noVoters) / len(feature) * noEntropy)
    return featureEntropy - firstValueEntropy - secondValueEntropy

columnHeader = ['output', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                'x16']
dataSet = pd.read_csv('house-votes-84.data.txt', header=None)

dataSet.columns = columnHeader
dataSet = dataSet.loc[:(50 / 100) * dataSet.shape[0], 'output':'x16']  # 25%
# dataset = dataset.loc[:1, 'out':'x16']
for index, row in dataSet.iterrows():
    tmp = np.array(row)
    yCount = 0
    nCount = 0
    tmpIndexes = []

    for i in range(len(tmp)):
        if tmp[i] == 'y':
            yCount += 1
        elif tmp[i] == 'n':
            nCount += 1
        elif tmp[i] == '?':
            tmpIndexes.append(i)

    for i in tmpIndexes:
        if yCount >= nCount:
            tmp[i] = 'y'
        else:
            tmp[i] = 'n'
    dataSet.loc[index] = tmp

root=Node()
completeTree(root,dataSet)

print("7mada")
