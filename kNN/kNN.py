#coding:utf-8

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator

def createDataSet():
	group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistance = sqDiffMat.sum(axis=1)
	distances = sqDistance**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	arrOLines = fr.readlines()
	numberOfLines = len(arrOLines)
	returnMat = np.zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	for line in arrOLines:
		line = line.strip()
		listPromLine = line.split('\t')
		returnMat[index:] = listPromLine[0:3]
		if listPromLine[-1] == 'didntLike':
			classLabelVector.append(1)
		elif listPromLine[-1] == 'smallDoses':
			classLabelVector.append(2)
		elif listPromLine[-1] == 'largeDoses':
			classLabelVector.append(3)
		index += 1
	return returnMat,classLabelVector

"""
Description: Normalize the data matrix

Parameters:
	dataSet - Feature matrix

Returns:
	normalDataSet - Normalized feature matrix
	ranges - Ranges of min/max values
	minValue - minimum value

"""

def autoNorm(dataSet):
	minValue = dataSet.min(0)
	maxValue = dataSet.max(0)

	ranges = maxValue - minValue
	normalDataSet = np.zeros(np.shape(dataSet))
	m = dataSet.shape[0]
	normalDataSet = dataSet - np.tile(minValue, (m, 1))
	normalDataSet = normalDataSet / np.tile(ranges, (m, 1))
	return normalDataSet, ranges, minValue
"""
description: Test 
	using the first 10% as test cases and the rest as feature matrix, k is 4

Parameters: None

Returns: None
"""

def datingClassTest():
	#测试数据
	filename = 'datingTestSet.txt'
	#测试数据转换为特征矩阵和标签数组
	datingDataMat,datingLabels = file2matrix(filename)
	#取百分之10作为测试样本
	hoRatio = 0.1
	#Normalize data matrix
	normalMat,ranges,minValue = autoNorm(datingDataMat)
	#normalMat的行数
	m = normalMat.shape[0]
	#测试数据的个数
	numOfTestVecs = int(m*hoRatio)
	#错误数
	errorCount = 0.0
	for i in range(numOfTestVecs):
		label = classify0(normalMat[i,:], normalMat[numOfTestVecs:m,:], datingLabels[numOfTestVecs:m], 4)
		print("classify result:%s\treal class:%d" % (label, datingLabels[i]))
		if label != datingLabels[i]:
			errorCount += 1
	print "error rate:%f%%" % (errorCount / float(numOfTestVecs) * 100)


def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentageVideoGame = float(input('percenage of the time spent playing video game:'))
	ffMiles = float(input('frequent flier miles earned per year:'))
	iceCream = float(input('liters of ice cream consumed per year:'))
	datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
	classifyResult = classify0(np.array([ffMiles, percentageVideoGame, iceCream]), datingDataMat, datingLabels, 4)
	print("You will probably %s this guy." % (resultList[classifyResult-1]))


"""
函数说明：可视化数据

Parameters:
	datingDataMat - 特征矩阵
	datingLabels - 分类标签
Returns:
	无
"""

def showDatas(datingDataMat, datingLabels):
	#设置汉字格式
	font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
	#将画布分为2行2列， 不共享x轴和y轴
	fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13,8))
	LabelsColors = []
	for label in datingLabels:
		if label == 1:
			LabelsColors.append("black")
		elif label == 2:
			LabelsColors.append("orange")
		elif label == 3:
			LabelsColors.append("red")

	#绘制第一幅图,x轴为每年获得的飞行里程数，y轴为玩游戏的时间占比
	axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors, s=15, alpha=0.5)
	axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', FontProperties=font)
	axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
	axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
	plt.setp(axs0_title_text, size=9, weight='bold', color='red')
	plt.setp(axs0_xlabel_text, size=7, weight='bold', color='red')
	plt.setp(axs0_ylabel_text, size=7, weight='bold', color='red')

	#绘制第二幅图，x轴为每年获得的飞行里程数，y轴为吃冰激凌的量
	axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors, s=15, alpha=0.5)
	axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与消耗的冰激凌', FontProperties=font)
	axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
	axs1_ylabel_text = axs[0][1].set_ylabel(u'消耗的冰激凌', FontProperties=font)
	plt.setp(axs1_title_text, size=9, weight='bold', color='red')
	plt.setp(axs1_xlabel_text, size=7, weight='bold', color='red')
	plt.setp(axs1_ylabel_text, size=7, weight='bold', color='red')

	#绘制第三张图,x轴为玩游戏的时间，y轴为冰激凌的消耗量
	axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors, s=15, alpha=0.5)
	axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间与消耗的冰激凌', FontProperties=font)
	axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间', FontProperties=font)
	axs2_ylabel_text = axs[1][0].set_ylabel(u'消耗的冰激凌', FontProperties=font)
	plt.setp(axs2_title_text, size=9, weight='bold', color='red')
	plt.setp(axs2_xlabel_text, size=7, weight='bold', color='red')
	plt.setp(axs2_ylabel_text, size=7, weight='bold', color='red')	


	#legends
	didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
	smallDoses = mlines.Line2D([], [], color='orange', marker='.',
					  markersize=6, label='smallDoses')
	largeDoses = mlines.Line2D([], [], color='red', marker='.',
		              markersize=6, label='largeDoses')

	axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
	axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
	axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

	plt.show()