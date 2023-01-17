import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

f = open("5year.arff", 'r')
# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

answers = {} # Your answers
def accuracy(predictions, y):
    right_num = 0
    for i in range(len(predictions)):
        if predictions[i] == y[i]:
            right_num += 1
    return right_num / len(predictions)
def BER(predictions, y):
    #FPR = FP / (TN + FP）
    #FNR = FN / (FN + TP)
    #BER = 1/2 (FPR + FNR)
    FP = 0
    TN = 0
    FN = 0
    TP = 0
    for i in range(len(predictions)):
        if predictions[i] == True and y[i] == True:
            TP += 1
        elif predictions[i] == True and y[i] == False:
            FP += 1
        elif predictions[i] == False and y[i] == True:
            FN += 1
        else:
            TN += 1
    FPR = FP / (TN + FP)
    FNR = FN / (FN + TP)
    BER = (FPR + FNR) / 2
    return BER
### Question 1
mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)
pred = mod.predict(X)
acc1 = accuracy(pred,y)
ber1 = BER(pred,y)
answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate
assertFloatList(answers['Q1'], 2)

### Question 2
mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)
pred = mod.predict(X)
acc2 = accuracy(pred,y)
ber2 = BER(pred,y)
answers['Q2'] = [acc2, ber2]
assertFloatList(answers['Q2'], 2)

### Question 3
random.seed(3)
random.shuffle(dataset)
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]
len(Xtrain), len(Xvalid), len(Xtest)
mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(Xtrain,ytrain)
pred_train = mod.predict(Xtrain)
pred_valid = mod.predict(Xvalid)
pred_test = mod.predict(Xtest)
berTrain = BER(pred_train,ytrain)
berValid = BER(pred_valid,yvalid)
berTest = BER(pred_test,ytest)
answers['Q3'] = [berTrain, berValid, berTest]
assertFloatList(answers['Q3'], 3)

### Question 4
berList = []
C_value = 10**(-4)
for i in range(9):
    mod = linear_model.LogisticRegression(C=C_value,class_weight='balanced')
    mod.fit(Xtrain,ytrain)
    pred_valid = mod.predict(Xvalid)
    berValid = BER(pred_valid,yvalid)
    berList.append(berValid)
    C_value = C_value * 10
answers['Q4'] = berList
assertFloatList(answers['Q4'], 9)

### Question 5
ber_best = berList[0]
C_value_index = 0
for i in range(len(berList)):
    if berList[i] < ber_best:
        ber_best = berList[i]
        C_value_index = i
bestC = 10**(-4 + C_value_index)
mod = linear_model.LogisticRegression(C=bestC,class_weight='balanced')
mod.fit(Xtrain,ytrain)
pred_test = mod.predict(Xtest)
ber5 = BER(pred_test,ytest)
answers['Q5'] = [bestC, ber5]
assertFloatList(answers['Q5'], 2)

### Question 6
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))
dataTrain = dataset[:9000]
dataTest = dataset[9000:]
# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair
for d in dataTrain:
    user,item = d['user_id'],d['book_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    ratingDict[item] = d['book_id']
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom

def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i:
            continue
        sim = Jaccard(users,usersPerItem[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:N]
answers['Q6'] = mostSimilar('2767052', 10)
assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)

### Question 7
for d in dataTrain:
    user,item = d['user_id'],d['book_id']
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)
ratingMean = sum([d['rating'] for d in dataTrain]) / len(dataTrain)

def get_average_r(item):
    sum_r = 0
    for d in reviewsPerItem[item]:
        sum_r += d['rating']
    average_r = sum_r / len(reviewsPerItem)
    return average_r

def predictRating(user,item):
    ratings = []
    similarities = []
    average_i_r = get_average_r(item)
    average_j_list = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item:
            continue
        ratings.append(d['rating'])
        average_j = get_average_r(i2)
        average_j_list.append(average_j)
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [((x - z) * y) for x,z,y in zip(ratings,average_j_list,similarities)]
        return average_i_r + sum(weightedRatings)/sum(similarities)
    else:
        return ratingMean
def MSE(predictions,labels):
    differences = [(x - y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

cfPredictions = [predictRating(d['user_id'],d['book_id']) for d in dataTest]
labels = [d['rating'] for d in dataTest]
mse7 = MSE(cfPredictions,labels)
answers['Q7'] = mse7
assertFloat(answers['Q7'])

### Question 8
def get_average_r_2(user):
    sum_r = 0
    for d in reviewsPerUser[user]:
        sum_r += d['rating']
    average_r = sum_r / len(reviewsPerUser)
    return average_r
def predictRating_2(user,item):
    ratings = []
    similarities = []
    average_i_r_2 = get_average_r(user)
    average_j_list = []
    for d in reviewsPerItem[item]:
        i2 = d['user_id']
        if i2 == user:
            continue
        ratings.append(d['rating'])
        average_j = get_average_r_2(i2)
        average_j_list.append(average_j)
        similarities.append(Jaccard(itemsPerUser[user],itemsPerUser[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [((x - z) * y) for x,z,y in zip(ratings,average_j_list,similarities)]
        return average_i_r_2 + sum(weightedRatings)/sum(similarities)
    else:
        return ratingMean
cfPredictions = [predictRating_2(d['user_id'],d['book_id']) for d in dataTest]
labels = [d['rating'] for d in dataTest]
mse8 = MSE(cfPredictions,labels)
answers['Q8'] = mse8
assertFloat(answers['Q8'])

f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()

