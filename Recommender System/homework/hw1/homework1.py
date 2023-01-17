import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import numpy
import random
import gzip
import math
import scipy
import sklearn

def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

len(dataset)

answers = {} # Put your answers to each question in this dictionary

dataset[0]
### Question 1
def feature(datum):
    # your implementation
    feat = datum['review_text'].count("!")
    return [1] + [feat]
X = [feature(d) for d in dataset]
Y = [d["rating"]for d in dataset]
X = numpy.matrix(X)
Y = numpy.matrix(Y)
result = numpy.linalg.inv(X.T*X)*X.T*Y.T
result = numpy.matrix.tolist(result)
theta0 = result[0][0]
theta1 = result[1][0]

square_sum = 0
X = [d['review_text'].count("!") for d in dataset]
Y = [d["rating"]for d in dataset]

for i in range(len(Y)):
    square_sum += (theta0 + X[i]*theta1 - Y[i])**2
mse = square_sum / len(Y)
answers['Q1'] = [theta0, theta1, mse]
assertFloatList(answers['Q1'], 3)

### Question 2
def feature(datum):
    feat_1 = len(datum["review_text"])
    feat_2 = datum['review_text'].count("!")
    return [1] + [feat_1,feat_2]
X = [feature(d) for d in dataset]
Y = [d["rating"]for d in dataset]
X = numpy.matrix(X)
Y = numpy.matrix(Y)
result = numpy.linalg.inv(X.T*X)*X.T*Y.T
result = numpy.matrix.tolist(result)
theta0 = result[0][0]
theta1 = result[1][0]
theta2 = result[2][0]

square_sum = 0
X_1 = [len(d["review_text"]) for d in dataset]
X_2 = [d['review_text'].count("!") for d in dataset]
Y = [d["rating"]for d in dataset]
for i in range(len(Y)):
    square_sum += (theta0 + X_1[i]*theta1 + X_2[i]*theta2 - Y[i])**2
mse = square_sum / len(Y)
answers['Q2'] = [theta0, theta1, theta2, mse]
assertFloatList(answers['Q2'], 4)

### Question 3
def feature_2(datum,deg):
    feat_list = []
    for j in range(deg):
        feat = datum['review_text'].count("!") ** (j + 1)
        feat_list.append(feat)
    return [1] + feat_list
mses = []
for i in range(1,6):
    degree = i
    X = [feature_2(d,i) for d in dataset]
    Y = [d["rating"]for d in dataset]
    lr = linear_model.LinearRegression()
    lr.fit(X,Y)
    Y_pre = lr.predict(X)
    mse = sklearn.metrics.mean_squared_error(Y,Y_pre)
    mses.append(mse)
answers['Q3'] = mses
assertFloatList(answers['Q3'], 5)# List of length 5
### Question 4
mses_list = []
for i in range(1,6):
    train_dataset = dataset[:int(len(dataset)/2)]
    test_dataset = dataset[int(len(dataset)/2):]
    X_train = [feature_2(d,i) for d in train_dataset]
    Y_train = [d["rating"]for d in train_dataset]
    lr = linear_model.LinearRegression()
    lr.fit(X_train, Y_train)
    X_test = [feature_2(d,i) for d in test_dataset]
    Y_test = [d["rating"] for d in test_dataset]
    Y_pre = lr.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test, Y_pre)
    mses_list.append(mse)
answers['Q4'] = mses
assertFloatList(answers['Q4'], 5)

### Question 5
train_dataset = dataset[:int(len(dataset) / 2)]
test_dataset = dataset[int(len(dataset) / 2):]
Y_train = [d["rating"] for d in train_dataset]
Y_test = [d["rating"] for d in test_dataset]
Y_train.sort()
theta0 = numpy.median(Y_train)
total_sum = 0
for i in range(len(Y_test)):
    total_sum += abs(Y_test[i] - theta0)
mae = total_sum / len(Y_test)
answers['Q5'] = mae
assertFloat(answers['Q5'])

### Question 6
f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))
len(dataset)


X = [[b['review/text'].count("!")] for b in dataset]
Y = [1 if b['user/gender'] == 'Female' else 0 for b in dataset]
mod = linear_model.LogisticRegression(C=1.0)
mod.fit(X, Y)
predictions = mod.predict(X)
TP = 0
TN = 0
FP = 0
FN = 0
BER = 0
for i in range(len(Y)):
    if predictions[i] == 1 and dataset[i]['user/gender'] == "Female":
        TP += 1
    elif predictions[i] == 0 and dataset[i]['user/gender'] == "Female":
        FN += 1
    elif predictions[i] == 1 and dataset[i]['user/gender'] == "Male":
        FP += 1
    else:
        TN += 1
FPR = FP / (FP + TN)
FNR = FN / (TP + FN)
BER = (FPR + FNR) / 2
answers['Q6'] = [TP, TN, FP, FN, BER]
assertFloatList(answers['Q6'], 5)

### Question 7
X = [[b['review/text'].count("!")] for b in dataset]
Y = [1 if b['user/gender'] == 'Female' else 0 for b in dataset]
mod = linear_model.LogisticRegression(C=1.0,class_weight='balanced')
mod.fit(X, Y)
predictions = mod.predict(X)
TP = 0
TN = 0
FP = 0
FN = 0
BER = 0
for i in range(len(Y)):
    if predictions[i] == 1 and dataset[i]['user/gender'] == "Female":
        TP += 1
    elif predictions[i] == 0 and dataset[i]['user/gender'] == "Female":
        FN += 1
    elif predictions[i] == 1 and dataset[i]['user/gender'] == "Male":
        FP += 1
    else:
        TN += 1
FPR = FP / (FP + TN)
FNR = FN / (TP + FN)
BER = (FPR + FNR) / 2
answers["Q7"] = [TP, TN, FP, FN, BER]
assertFloatList(answers['Q7'], 5)

### Question 8
scores = mod.decision_function(X)
scoreslabels = list(zip(scores,Y))
sortedlabels = [a[1] for a in scoreslabels]

p1 = sum(sortedlabels[:1]) / 1
p2 = sum(sortedlabels[:10]) / 10
p3 = sum(sortedlabels[:100]) / 100
p4 = sum(sortedlabels[:1000]) / 1000
p5 = sum(sortedlabels[:10000]) / 10000
precisionList = [p1,p2,p3,p4,p5]
answers['Q8'] = precisionList
assertFloatList(answers['Q8'], 5) #List of five floats

f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()

