import numpy as np
import pandas as pd


eps = 1E-14

def calculateEntropyC(dataSet):
    numEntropy = len(dataSet)
    labelCounts = {}

    for data in dataSet:
        currentLabel = data[-1]
        
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    
    totalEntropy = 0.0

    for key in labelCounts:
        prob = labelCounts[key] / numEntropy
        totalEntropy -= prob * np.log2(prob)
    
    return totalEntropy

def calculateEntropyS(y):
    labels, counts = np.unique(y, return_counts = True)
    counts.dtype = 'float64'
    probs = counts / np.sum(counts)
    return -(probs * np.log2(probs)).sum()

def calculateConditionalEntropy(xf, y):
    xf_red = np.unique(xf)
    ConditionalEntropy = 0.0
    m = y.shape[0]
    n = len(xf[0])
    for xf_one in xf_red:
        mask = xf == xf_one
        y_one = y[mask]
        prob_one = float(y_one.shape[0] / m)
        ConditionalEntropy += prob_one * calculateEntropyS(y_one)
    return ConditionalEntropy

def calculateFeatureEntropy(xf, y ):
    xf_red = np.unique(xf)
    featureEntropy = 0.0
    m = y.shape[0]
    n = len(xf[0])
    for xf_one in xf_red:
        mask = xf == xf_one
        y_one = y[mask]
        prob_one = float(y_one.shape[0] / m)
        featureEntropy -= prob_one * np.log2(prob_one)
    return featureEntropy   
            
def chooseBestFeature(x, y):
    Entropy = calculateEntropyS(y)
    bestFeature = 0
    maxRatio = 0
    for feature in range(x.shape[1]):
        fetval = x[:, feature]

        infoGain = Entropy - calculateConditionalEntropy(fetval, y)
        valEntropy = calculateFeatureEntropy(fetval, y)
        infoGainRatio = float(infoGain) / (valEntropy + eps)

        if(infoGainRatio > maxRatio):
            maxRatio = infoGainRatio
            bestFeature = feature
    return bestFeature, maxRatio

def majority(classList):
    classcount = {}
    for vote in classList:
        if vote not in classList.keys():
            classcount[vote] = 0
        classcount[vote] += 1
    
    return max(classcount)

def splitdataset(xtrain, current_train, value):
    subdataset = []
    for train in xtrain:
        if train[current_train] == value:
            ruducedtrain = np.hstack((xtrain[:current_train], xtrain[current_train + 1: ]))
            subdataset.append(ruducedtrain)
    return subdataset

#建树

def createTree(x, label):
    classList = [exp[-1] for exp in x]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if (len(x[0]) == 1):
        return majority(classList)

    bestFeature = chooseBestFeature(x[:, :-1], x[:, -1])
    bestLabel = label[bestFeature]
    tree = {bestLabel:{}}
    del(label[bestFeature])
    featureVal = [exp[bestFeature] for exp in x]
    uniqueVal = set(featureVal)
    for val in uniqueVal:
        subLabel = label[:]
        tree[bestLabel][val] = createTree(splitdataset(x, bestFeature, val), subLabel)
    return tree