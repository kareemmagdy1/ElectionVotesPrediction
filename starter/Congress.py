import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from starter.Tree import Tree
from starter.Node import Node

def predict(row,node):
    if(len(node.children)==0):
        return node.name
    index=columnHeader.index(node.name)
    value=row[index-1]
    for index,childValues in enumerate(node.possibleValues):
        if(value==childValues):
           return predict(row,node.children[index])
            # node=node.children[index]

def calculateTreeSize(node):
    nodesNumber=0
    for child in node.children:
         nodesNumber+=calculateTreeSize(child)
    return nodesNumber+len(node.children)

def calculateAccuracyoFTree(root, dataSet):
    expectedAnswers=[]
    predictedAnswers=[]
    for index, row in dataSet.iterrows():
        tmp = np.array(row)
        expectedAnswers.append(tmp[0])
        temp=tmp[1:]
        predictedAnswers.append(predict(temp,root))

    error=0
    for i in range(len(predictedAnswers)):
        if(predictedAnswers[i]!=expectedAnswers[i]):
            error+=1
    return error


def part1():
    dataSet = pd.read_csv('house-votes-84.data.txt', header=None)

    dataSet.columns = columnHeader
    trainingDataSet = dataSet.loc[:(25 / 100) * dataSet.shape[0], 'output':'x16']
    testingDataSet=dataSet.loc[(25 / 100) * dataSet.shape[0]:, 'output':'x16':]
    root=Node()
    completeTree(root, trainingDataSet)
    print("TreeSize: ",calculateTreeSize(root)+1)
    print("Training Error: ",calculateAccuracyoFTree(root,trainingDataSet))
    print("Training Error: ",calculateAccuracyoFTree(root,testingDataSet))


def getPossibleFeatureValues(feature):
    possibleValues=[]
    for x in feature:
        if not(x in possibleValues):
            possibleValues.append(x)
    return possibleValues


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


def completeTree(node,dataset):
    output=dataset['output'].values.tolist()
    dataSetEntropy=getFeatureEntropy(output)
    if(dataSetEntropy==0):
        node.name=output[0]
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

    node.possibleValues=possbileValues
    for index,value in enumerate(possbileValues):
        node.children.append(Node())
        completeTree(node.children[index],dataset[dataset[node.name]==value])


def getFeatureEntropy(outputs):
    democrates = 0
    republicans = 0
    for i in range(len(outputs)):
        if (outputs[i] == 'democrat'):
            democrates += 1
        else:
            republicans += 1
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

columnHeader = ['output', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                'x16']
# dataSet = pd.read_csv('house-votes-84.data.txt', header=None)
#
# dataSet.columns = columnHeader
# dataSet = dataSet.loc[:(70 / 100) * dataSet.shape[0], 'output':'x16']  # 25%
# for index, row in dataSet.iterrows():
#     tmp = np.array(row)
#     yCount = 0
#     nCount = 0
#     tmpIndexes = []
#
#     for i in range(len(tmp)):
#         if tmp[i] == 'y':
#             yCount += 1
#         elif tmp[i] == 'n':
#             nCount += 1
#         elif tmp[i] == '?':
#             tmpIndexes.append(i)
#
#     for i in tmpIndexes:
#         if yCount >= nCount:
#             tmp[i] = 'y'
#         else:
#             tmp[i] = 'n'
#     dataSet.loc[index] = tmp

# root=Node()
#
# completeTree(root,dataSet)
#
#
# arr=['n','y','y','n','y','y','n','n','n','n','n','n','y',"y","y","y"]

# print(calculateTreeSize(root))
# print(predict(arr,root))
# print("done")



part1()