#!/usr/bin/python
import numpy as np
from math import log
data = np.genfromtxt('data/breast-cancer-wisconsin.data.txt', delimiter=',')
X_train = data[:490]
X_valid = data[490:560]
X_test = data[560:]
def countResult(rows): #count the number of each result(output)
    results = {}
    for row in rows:
        r = row[len(row)-1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results #return a dic_ex.{2:295, 4:195}


def entropy(rows):
    results = countResult(rows)
    #Get entropy H(x)
    ent = 0.0
    for r in results.keys(): #2 and 4
        p = float(results[r])/len(rows) #the probability of the result=2 or 4
        ent = ent - p*(log(p)/log(2)) #count H(x)
    return ent

class treeNode:
    def __init__(self, column=-1, value=None, results=None, trueBranch=None, falseBranch=None):
        self.column = column
        self.value = value
        self.results = results
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch


def divideset(rows, column, value): #seperate the rows by the results
    set1 = [row for row in rows if row[column] >= value]
    set2 = [row for row in rows if row[column] < value]
    return(set1, set2)


def buildtree(rows):
    if len(rows) == 0:
        return treeNode() #rootNode
    best_gain = 0.0
    best_criteria = None
    best_sets = None


    for column in range(1, len(rows[0])-1):
        column_values = {}
        for row in rows:
            column_values[row[column]] = 1  # store every value of a column

        # find the highest gain of a column to separate data to two sets
        for value in column_values.keys():
            (set1, set2) = divideset(rows, column, value)
            #gain information
            p = float(len(set1)) / len(rows)
            gain = entropy(rows) - p * entropy(set1) - (1 - p) * entropy(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (column, value)
                best_sets = (set1, set2)
    if best_gain > 0: # build branches
        return treeNode(column=best_criteria[0], value=best_criteria[1],
                            trueBranch=buildtree(best_sets[0]), falseBranch=buildtree(best_sets[1]))
    else:
        return treeNode(results=countResult(rows))

def classify(input, tree):

    if tree.results!= None:
        return tree.results
    else:
        if input[tree.column] >= tree.value:
            branch = tree.trueBranch
        else:
            branch = tree.falseBranch

        return classify(input, branch)


divideset(X_train, 10, 4)
tree = buildtree(X_train)

#validation
correct = 0
for i in range(len(X_valid)):

    y_valid = X_valid[i][-1]
    y_hat = classify(X_valid[i][:10], tree)
    if y_valid in y_hat:
        correct += 1
print("Accuracy of valid:", correct/len(X_valid))

#test
correct = 0
for i in range(len(X_test)):

    y_test = X_test[i][-1]
    y_hat = classify(X_test[i][:10], tree)
    if y_test in y_hat:
        correct += 1
    #print(y_test, y_hat)
print("Accuracy of test:", correct/len(X_test))