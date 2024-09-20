#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
import math
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1/(1 + exp(gamma))
    else:
        return 1/(1 + exp(-gamma))

def gradient_descent(data, w, b):
    # Perform Batch Gradient Descent
    numvars = len(data[0][0])
    grad_w = [0.0] * numvars
    grad_b = 0.0
    for x, y in data:
        #p: prediction value
        p = sum([w[i] * x[i] for i in range(numvars)]) + b
        for i in range(numvars):
            grad_w[i] += y*x[i]*sigmoid(-y*p)
        grad_b += y*sigmoid(-y*p)
    
    return(grad_w, grad_b)




# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    b = 0.0
    w = [0.0] * numvars
    
    for _ in range(MAX_ITERS):
        grad_w, grad_b = gradient_descent(data, w, b)

        # Add regularizatioin and update the weight
        for i in range(numvars):
            w[i] += eta*(grad_w[i] - l2_reg_weight*w[i])
        b += eta*grad_b       
    return (w,b)



# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model
    prediction = sum([w[i] * x[i] for i in range(len(w))]) + b
    return sigmoid(prediction) # This is an random probability, fix this according to your solution


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])