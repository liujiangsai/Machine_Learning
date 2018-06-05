# -*- coding: UTF-8 -*-
import numpy as np
from math import log
import random

"""
Description: ceate test dataset

Parameters: None

Returns:

"""
def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],				
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]   																#class vector, 1 means bad word
	return postingList,classVec	

"""

Returns:
	vocabSet - list of unique words

"""

def createVocabList(dataSet):
	vocabSet = set([])
	for data in dataSet:
		vocabSet = vocabSet | set(data)
	return list(vocabSet)

def setOfWords2Vec(vocabList, inputVec):
	returnVec = [0]*len(vocabList)
	for word in inputVec:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
	return returnVec

def bagOfWords2Vec(vocabList, inputVec):
	returnVec = [0]*len(vocabList)
	for i in range(len(inputVec)):
		word = inputVec[i]
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

def trainNB(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p1Num = np.ones(numWords)
	p0Num = np.ones(numWords)
	p1Denom = 2.0
	p0Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vec = np.log(p1Num / p1Denom)
	p0Vec = np.log(p0Num / p0Denom)
	return p0Vec,p1Vec,pAbusive

def classifyNB(inputMatrix, p0Vec, p1Vec, pAbusive):
	p0 = sum(inputMatrix*p0Vec) + log(1-pAbusive)
	p1 = sum(inputMatrix*p1Vec) + log(pAbusive)
	if p0 > p1:
		return 0
	else:
		return 1

def testingNB():
	listOfPosts,listCalasses = loadDataSet()
	myVocabList = createVocabList(listOfPosts)
	trainMatrix = []
	for postinDoc in listOfPosts:
		trainMatrix.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V,p1V,pAb = trainNB(np.array(trainMatrix), np.array(listCalasses))
	testEntry = ['love','dalmation']
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	print testEntry,'classified as:',classifyNB(thisDoc, p0V, p1V, pAb)

def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
	docList = []; classList = []
	for i in range(1,26):
		wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
		docList.append(wordList)
		classList.append(1)
		wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
		docList.append(wordList)
		classList.append(0)
	myVocabList = createVocabList(docList)
	trainSet = list(range(50));testSet = []
	for i in range(10):
		randomIndex = int(random.uniform(0,len(trainSet)))
		testSet.append(trainSet[randomIndex])
		del(trainSet[randomIndex])
	trainMatrix = []; trainCategory = []
	for index in trainSet:
		trainMatrix.append(setOfWords2Vec(myVocabList, docList[index]))
		trainCategory.append(classList[index])
	p0V,p1V,pAbusive = trainNB(np.array(trainMatrix), np.array(trainCategory))
	errorCount = 0
	for index in testSet:
		testMatrix = setOfWords2Vec(myVocabList, docList[index])
		testClass = classList[index]
		if classifyNB(np.array(testMatrix), p0V, p1V, pAbusive) != testClass:
			print 'wrong index:' + str(index)
			print 'error text:',docList[index]
			errorCount += 1
	print 'total error rate %.2f%%' % (float(errorCount)/len(testSet)*100)


if __name__ == '__main__':
	for i in range(10):
		spamTest()		