import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from starter.Tree import Tree
from starter.Node import Node


def predict(row, node):
    if (len(node.children) == 0):
        return node.name
    index = columnHeader.index(node.name)
    value = row[index - 1]
    for index, childValues in enumerate(node.possibleValues):
        if (value == childValues):
            return predict(row, node.children[index])
            # node=node.children[index]


def calculateTreeSize(node):
    nodesNumber = 0
    for child in node.children:
        nodesNumber += calculateTreeSize(child)
    return nodesNumber + len(node.children)


def calculateAccuracyoFTree(root, dataSet):
    expectedAnswers = []
    predictedAnswers = []
    for index, row in dataSet.iterrows():
        tmp = np.array(row)
        expectedAnswers.append(tmp[0])
        temp = tmp[1:]
        predictedAnswers.append(predict(temp, root))

    error = 0
    for i in range(len(predictedAnswers)):
        if (predictedAnswers[i] != expectedAnswers[i]):
            error += 1
    return error


def part2():
    dataset = setupPart2DataSet()
    for i in range(5):
        curr=dataset
        trees_sizes=[]
        accuracies=[]
        percent = 30 + i * 10
        print('----------------------------------------------------')
        print('training data with ',percent,'%')
        print('----------------------------------------------------')
        for j in range(5):
            # print('Iteration #',j+1)
            train = curr.sample(frac=percent / 100)
            test = curr.drop(train.index)
            root = Node()
            completeTree(root, train)
            # print("TreeSize: ", calculateTreeSize(root) + 1)
            # print("Training Accuracy: ", (len(train) - calculateAccuracyoFTree(root, train)) / len(train))
            acc=(len(test) - calculateAccuracyoFTree(root, test)) / len(test)
            accuracies.append(acc)
            # print("Testing Accuracy: ", acc)
            singleTreeSize=calculateTreeSize(root) + 1
            trees_sizes.append(singleTreeSize)
            # print("Tree size: ", singleTreeSize)
            root = None

        print("min acc: ",min(accuracies))
        print("max acc: ",max(accuracies))
        print('Min Tree Size: ', min(trees_sizes))
        print("Mean Tree Size: ",np.sum(trees_sizes)/len(trees_sizes))
        print('Max Tree Size: ', max(trees_sizes))

def setupPart2DataSet():
    dataSet = pd.read_csv('house-votes-84.data.txt', header=None)
    dataSet.columns = columnHeader
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
    return dataSet


def part1():
    dataSet = pd.read_csv('house-votes-84.data.txt', header=None)
    dataSet.columns = columnHeader
    for i in range(5):
        print('#iteration ', i + 1)
        curr = dataSet
        train = curr.sample(frac=20 / 100)
        test = curr.drop(train.index)
        root = Node()
        completeTree(root, train)
        print("TreeSize: ", calculateTreeSize(root) + 1)
        print("Training Accuracy: ", (len(train) - calculateAccuracyoFTree(root, train)) / len(train))
        print("Testing Accuracy: ", (len(test) - calculateAccuracyoFTree(root, test)) / len(test))
        print("Tree size: ", calculateTreeSize(root) + 1)
        root = None
        print('----------------------------------------------------')


def getPossibleFeatureValues(feature):
    possibleValues = []
    for x in feature:
        if not (x in possibleValues):
            possibleValues.append(x)
    return possibleValues


def calculateGainForAttribuite(feature, featureEntropy, outputs):
    valuesVoters = []
    valuesEntropy = []
    possibleValues = getPossibleFeatureValues(feature)
    for _ in range(len(possibleValues)):
        valuesVoters.append([])
        valuesEntropy.append(0)

    for index in range(len(feature)):
        for i, value in enumerate(possibleValues):
            if (feature[index] == value):
                valuesVoters[i].append(index)
    for index, voters in enumerate(valuesVoters):
        if (len(voters) != 0):
            valuesEntropy[index] = getValueEntropy(voters, outputs)
    gain = featureEntropy
    for index, value in enumerate(valuesVoters):
        gain -= (len(value) / len(feature) * valuesEntropy[index])
    return gain


def calculateEntropy(numberOfYes, numberOfNo):
    if (numberOfNo == 0 and numberOfYes == 0):
        return 0
    sum = numberOfYes + numberOfNo
    y = 0
    k = 0
    x = -1 * numberOfYes / sum
    if (x != 0):
        y = math.log(float(numberOfYes) / float(sum), 2)
    z = numberOfNo / sum
    if (z != 0):
        k = math.log(float(numberOfNo) / float(sum), 2)
    return (x * y - 1 * z * k)


def completeTree(node, dataset):
    output = dataset['output'].values.tolist()
    dataSetEntropy = getFeatureEntropy(output)
    if (dataSetEntropy == 0):
        node.name = output[0]
        return
    gainValuesPerFeature = []
    for element in columnHeader:
        if element == 'output':
            gainValuesPerFeature.append(0)
            continue
        feature = dataset[element].values.tolist()
        gainValuesPerFeature.append(
            calculateGainForAttribuite(feature, dataSetEntropy, dataset['output'].values.tolist()))
    maxGain = max(gainValuesPerFeature)
    maxIndex = gainValuesPerFeature.index(maxGain)
    node.name = columnHeader[maxIndex]
    possbileValues = getPossibleFeatureValues(dataset[columnHeader[maxIndex]].values.tolist())

    node.possibleValues = possbileValues
    for index, value in enumerate(possbileValues):
        node.children.append(Node())
        completeTree(node.children[index], dataset[dataset[node.name] == value])


def getFeatureEntropy(outputs):
    democrates = 0
    republicans = 0
    for i in range(len(outputs)):
        if (outputs[i] == 'democrat'):
            democrates += 1
        else:
            republicans += 1
    return calculateEntropy(democrates, republicans)


def getValueEntropy(valueVoters, outputs):
    democrates = 0
    republicans = 0
    for index in valueVoters:
        if (outputs[index] == 'democrat'):
            democrates += 1
        else:
            republicans += 1
    return calculateEntropy(democrates, republicans)


columnHeader = ['output', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14',
                'x15',
                'x16']

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
part2()
