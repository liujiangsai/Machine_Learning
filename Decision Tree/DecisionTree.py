# -*- coding:UTF-8 -*-

import numpy as np
from math import log
import operator
import pickle
import copy

"""
Description: 
	Calculate Shannon Entropy
Parameters:
	dataSet 
Returns:
	Shannon entropy
"""

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)						#返回数据集的行数
	labelCounts = {}								#保存标签到字典
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1				#提取标签数量并累加
	shannonEnt = 0.0
	for label in labelCounts:
		prob = float(labelCounts[label])/float(numEntries)	#选择该标签(Label)的概率
		shannonEnt -= prob * log(prob, 2)					#公式
	return shannonEnt

"""
函数说明:创建测试数据集

Parameters:
	无
Returns:
	dataSet - 数据集
	labels - 特征标签
"""
def createDataSet():
	dataSet = [[0, 0, 0, 0, 'no'],						#数据集
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
	labels = ['age', 'job', 'house', 'credit']		#特征标签
	return dataSet, labels 	

	"""
函数说明:按照给定特征划分数据集

Parameters:
	dataSet - 待划分的数据集
	axis - 划分数据集的特征
	value - 需要返回的特征的值
Returns:
	划分后的子集
"""
def splitDataSet(dataSet, axis, value):		
	resultSet = []							#返回划分后的数据集
	for data in dataSet:
		if data[axis] == value:
			reducedFeatureVec = data[:axis]
			reducedFeatureVec.extend(data[axis+1:])
			resultSet.append(reducedFeatureVec)
	return resultSet

"""
函数说明:选择最优特征

Parameters:
	dataSet - 数据集
Returns:
	bestFeature - 信息增益最大的(最优)特征的索引值
"""

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0])-1 			#特征数量
	baseEntropy = calcShannonEnt(dataSet)		#数据集的香农熵
	bestInfoGain = 0.0							#最优熵
	bestFeature = -1							#最优特征索引
	for i in range(numFeatures):
		uniqueValues = [example[i] for example in dataSet]		#取第i列所有特征
		uniqueValues = set(uniqueValues)
		newFeatureEntropy = 0.0
		for value in uniqueValues:
			subSet = splitDataSet(dataSet, i, value)			#划分子集
			prob = float(len(subSet))/float(len(dataSet))
			newFeatureEntropy += prob * calcShannonEnt(subSet)
		infoGain = baseEntropy - newFeatureEntropy
		print "index %d infoGain: %f" % (i, infoGain)
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature


"""
函数说明: 取出现次数最多的类别的名字

Parameters:
	classList - 所有类别

Returns:
	sortedClassCount[0][0] - 出现次数最多的类别
"""
def majorityCount(classList):
	labelCounts = {}
	for vote in classList:
		if vote not in labelCounts.keys():
			labelCounts[vote] = 0
		labelCounts[vote] += 1
	sortedClassCount = sorted(labelCounts.items(), operator.itemgetter(1), reverse=True)  
	return sortedClassCount[0][0]

"""
Description:
	create decision tree by given dataSet & label
Parameters:
	dataSet - a numpy array
	labelsx - list of label
Returns:
	myTree - decision tree
"""

def createTree(dataSet, labelsx):
	labels = copy.copy(labelsx)
	classList = [example[-1] for example in dataSet]
	print dataSet
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:									#遍历完所有特征时返回出现次数最多的类标签
		return majorityCnt(classList)
	bestFeature = chooseBestFeatureToSplit(dataSet)
	bestLabel = labels[bestFeature]
	myTree = {bestLabel:{}}
	del(labels[bestFeature])
	featureValues = [example[bestFeature] for example in dataSet]
	uniqueValues = set(featureValues)
	for value in uniqueValues:
		splitSet = splitDataSet(dataSet, bestFeature, value)
		myTree[bestLabel][value] = createTree(splitSet, labels)
	return myTree


import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	if parentPt == None:
		parentPt = centerPt
	createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
		va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

"""
"""
def plotMyTree(myTree, parentPt, nodeTxt):
	textStr = next(iter(myTree))
	numLeafs = getNumLeafs(myTree)
	centerPt = (plotMyTree.xOffset + ((float(numLeafs)+1)*1/plotMyTree.totalW)/2,plotMyTree.yOffset)
	if parentPt != None:
		plotMidText(centerPt, parentPt, nodeTxt)
	plotNode(textStr, centerPt, parentPt, decisionNode)
	subDict = myTree[textStr]
	plotMyTree.yOffset = plotMyTree.yOffset - 1.0/plotMyTree.totalD	
	for key in subDict.keys():
			if type(subDict[key]).__name__ == 'dict':
				plotMyTree(subDict[key], centerPt, key)
			else:
				plotMyTree.xOffset = plotMyTree.xOffset + 1.0/plotMyTree.totalW
				plotMidText((plotMyTree.xOffset, plotMyTree.yOffset), centerPt, key)
				plotNode(subDict[key], (plotMyTree.xOffset, plotMyTree.yOffset), centerPt, leafNode)
	plotMyTree.yOffset = plotMyTree.yOffset + 1.0/plotMyTree.totalD

def createPlot(myTree):
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	createPlot.ax1 = plt.subplot(111, frameon=False)

	plotMyTree.totalW = getNumLeafs(myTree)
	plotMyTree.totalD = getDepth(myTree)
	plotMyTree.xOffset = -0.5/float(plotMyTree.totalW)
	plotMyTree.yOffset = 1.0
	print plotMyTree.xOffset, plotMyTree.yOffset
	plotMyTree(myTree, None, '')

	# plotNode('a decision node', (0.5,0.1), (0.1, 0.5), decisionNode)
	# plotNode('a leaf node', (0.8,0.1), (0.3, 0.8), leafNode)
	plt.show()

def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]											#计算标注位置					
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def getTree():
	return {'house': {0: {'job': {0: 'no', 1: 'yes'}}, 1: {'credit': {0: 'yes', 1: 'no'}}}}


"""
Description: Get number of leaf node, to decide the length of X axis

Parameters:
	myTree - Decision Tree

Returns:
	numLeafs - number of leaf node
"""

def getNumLeafs(myTree):
	numLeafs = 0
	firstDict = myTree[myTree.keys()[0]]
	for key in firstDict.keys():
		if type(firstDict[key]).__name__ == 'dict':
			numLeafs += getNumLeafs(firstDict[key])
		else:
			numLeafs += 1
	return numLeafs

"""
Description: Get depth of the decision tree, to decide the height of Y axis

Parameters:
	myTree - Decision Tree

Returns:
	maxDepth - depth of decision tree
"""
def getDepth(myTree):
	maxDepth = 0
	firstDict = myTree[myTree.keys()[0]]
	for key in firstDict.keys():
		if type(firstDict[key]).__name__ == 'dict':
			thisDepth = 1 + getDepth(firstDict[key])	#recursive call, each time +1 (itself)
		else:
			thisDepth = 1
		if thisDepth > maxDepth:
			maxDepth = thisDepth
	return maxDepth

""""""
def classify(inputTree, featureLabels, testVec):
	firstStr = next(iter(inputTree))
	featureIndex = featureLabels.index(firstStr)
	secondDict = inputTree[firstStr]
	for key in secondDict.keys():
		if key == testVec[featureIndex]:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featureLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel

def testCase0():
	dataSet,labels = createDataSet()
	myTree = createTree(dataSet, labels)
	testVec = [2, 1, 0, 2]
	print classify(myTree, labels, testVec)
	createPlot(myTree)

def createDataSetLenses():
	dataSet = []
	f = open('lenses.txt')
	for line in f.readlines():
		dataSet.append(line.strip().split('\t'))
	f.close()
	labels = ['age', 'prescript', 'astigmatic', 'tearRate']
	return dataSet,labels
if __name__ == '__main__':
	dataSet,labels = createDataSetLenses()
	myTree = createTree(dataSet, labels)
	testVec = ['young','hyper','no','reduced']
	print classify(myTree, labels, testVec)
	createPlot(myTree)