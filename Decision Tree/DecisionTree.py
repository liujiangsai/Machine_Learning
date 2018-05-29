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
	createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
		va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot():
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	createPlot.ax1 = plt.subplot(111, frameon=False)
	plotNode('a decision node', (0.5,0.1), (0.1, 0.5), decisionNode)
	plotNode('a leaf node', (0.8,0.1), (0.3, 0.8), leafNode)
	plt.show()


if __name__ == '__main__':
	createPlot()