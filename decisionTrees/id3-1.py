#!/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #2
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
# Node class for the decision tree
import node
import math

train = None
varnames = None
test = None
testvarnames = None
root = None
labels = None
best_split_features = []

# Helper function computes entropy of Bernoulli distribution with
# parameter p
# p is the positive/total
# positives = sum(1 for row in p if row[-1] == 1)
# n = len(p)
def entropy(p):
    # >>>> YOUR CODE GOES HERE <<<<
    if p == 0 or p == 1:
        return 0
    return -p* math.log2(p) - (1-p)* math.log2(1-p)
    
    

# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    # >>>> YOUR CODE GOES HERE <<<<
    E_total = entropy(py/total)
    if pxi == 0 or pxi == total:
        return 0
    else:
        positive= py_pxi/pxi
        negative = (py-py_pxi)/(total-pxi)
    
    result = E_total - (pxi/total)*entropy(positive) - ((total-pxi)/total)*entropy(negative)
    return result

# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable

def count(data):
    left = sum(1 for line in data if line[-1] == 0)
    right = sum(1 for line in data if line[-1] == 1)
    if left < right:
        return 1
    else:
        return 0



def partition(data, feature):
    left = [line for line in data if line[feature] == 0] 
    right = [line for line in data if line[feature] == 1]  
    return left, right    

   
    


def best_split_feature(data,varnames):
    best_infogain = -1
    best_feature = None
    for i in range(len(varnames) - 1):
        origin_index = labels.index(varnames[i])
        # prepare py_pxi, pxi, py, total
        total = len(data)
        py = sum(1 for line in data if line[-1] == 1)
        left, right = partition(data, origin_index)
        pxi = len(right)
        py_pxi = sum(1 for line in right if line[-1] == 1)
        info_gain = infogain(py_pxi, pxi, py, total)
        if info_gain > best_infogain:
            best_infogain = info_gain
            best_feature = (i, origin_index, labels[origin_index])
    if best_feature is None or best_infogain == 0:
        return None, 0
    return best_feature, best_infogain



# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return a leaf predicting "1":
    default = count(data)
    if len(varnames) == 0:
        return node.Leaf(labels, default)
    if all(line[-1] == data[0][-1] for line in data):
        return node.Leaf(labels, data[0][-1])

    best_feature, best_info_gain = best_split_feature(data, varnames)
    if best_info_gain == 0 or best_feature is None:
        return node.Leaf(labels, default)
    
    best_feature_index, best_feature_origin_index,best_feature_name = best_feature
    best_split_features.append(best_feature_name)

    left_data, right_data = partition(data, best_feature_origin_index)
    new_varnames = [f for f in varnames if f != varnames[best_feature_index]]

    left_subtree = build_tree(left_data, new_varnames)
    right_subtree = build_tree(right_data, new_varnames)

    
    return node.Split(labels, best_feature_origin_index, left_subtree, right_subtree)


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    global labels
    (train, varnames) = read_data(trainS)
    labels = varnames
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
