#!/usr/bin/python
import numpy as np
import collections

stopList = np.genfromtxt("./stoplist.txt", delimiter="\t", dtype="str")
stopDict = collections.Counter()
for i in stopList:
    stopDict[i] = 1
trainX = np.genfromtxt("./traindata.txt", delimiter="\t", dtype="str")
trainY = np.genfromtxt("./trainlabels.txt", delimiter="\t", dtype="int")
testX = np.genfromtxt("./testdata.txt", delimiter="\t", dtype="str")
testY = np.genfromtxt("./testlabels.txt", delimiter="\t", dtype="int")

def replaceSystem(string):
    for char in string:
        if char in "~!@#$%^&*()[]{},+-|/?<>.;:0123456789":
            string = string.replace(char, '')
    return string

def split_line(text):
    text = replaceSystem(text)
    text = text.lower()
    words = text.split()
    words = removeStop(words)
    return words

def removeStop(string):
    i = len(string)
    j = 0
    while j<i:
        if string[j] in stopDict:
            del string[j]
            i -= 1
            j -= 1
        j += 1
    return string

def caculator(string, token, total, len):
    #print(string, total, len)
    # remove stop words and count how many words not in Dict
    state = 0
    for i in string.split():
    #for i in split_line(string):
        if i not in token:
            total += 1
            state = 1
    # remove stop words and count the percentage
    result = 1
    for i in string.split():
    #for i in split_line(string):
        if i in token:
            result = result * ((token[i]+state)/total)
        else:
            result = result * (1/total)

    return result * len

def countYNum(Y):
    for i in range(len(Y)):
        if Y[i] == 0:
            break
    return i+1

def makeDict(X):
    DictX = collections.Counter()
    for i in X:
        newLine = split_line(i)
        for j in newLine:
            DictX[j] += 1
    return DictX

# separate 0 and 1
One = countYNum(trainY)
trainXone = trainX[:One]  # 0~151
trainXzero = trainX[One:] # 152~321

# make dictionaries
tokenOne = makeDict(trainXone)
tokenZero = makeDict(trainXzero)
tokenTotal = makeDict(trainX)
#test training data
counter = 0
lenX = len(trainX)
Zero = lenX - One
for i in range(lenX):
    #count how many words in the dictionary
    p_one = caculator(trainX[i], tokenOne, len(tokenTotal), One/lenX)
    p_zero = caculator(trainX[i], tokenZero, len(tokenTotal), Zero/lenX)
    y_one = p_one / (p_one+p_zero)
    y_zero = p_zero / (p_one+p_zero)
    if y_one > y_zero and trainY[i]==1:
        counter += 1
    elif y_one < y_zero and trainY[i]==0:
        counter += 1
print("training accuracy", counter/lenX)

#test testing data
counter = 0
lentestX = len(testX)
#print(len(tokenOne), len(tokenZero))
for i in range(lentestX):
    #count how many words in the dictionary
    p_one = caculator(testX[i], tokenOne, len(tokenTotal), One/lenX)
    p_zero = caculator(testX[i], tokenZero, len(tokenTotal), Zero/lenX)
    y_one = p_one / (p_one+p_zero)
    y_zero = p_zero / (p_one+p_zero)
    if y_one > y_zero and testY[i]==1:
        counter += 1
    elif y_one < y_zero and testY[i]==0:
        counter += 1
print("testing accuracy", counter/lentestX)

file = open(r"output.txt", "w+")
file.write("traing accuracy" + str(counter/lenX) + '\n')
file.write("testing accuracy" + str(counter/lentestX))
file.close()