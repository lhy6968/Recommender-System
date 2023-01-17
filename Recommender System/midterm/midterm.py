import json
import gzip
import math
from collections import defaultdict
import numpy
from sklearn import linear_model
# This will suppress any warnings, comment out if you'd like to preserve them
import warnings
import sklearn
warnings.filterwarnings("ignore")
# Check formatting of submissions
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
answers = {}
f = open("spoilers.json.gz", 'r')
dataset = []
for l in f:
    d = eval(l)
    dataset.append(d)
f.close()

# A few utility data structures
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u, i = d['user_id'], d['book_id']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)

# Sort reviews per user by timestamp
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['timestamp'])

# Same for reviews per item
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['timestamp'])

# E.g. reviews for this user are sorted from earliest to most recent
[d['timestamp'] for d in reviewsPerUser['b0d7e561ca59e313b728dc30a5b1862e']]

### 1a
def get_all_rating_for_user(u):
    user_info = []
    rating = []
    for d in dataset:
        if d['user_id'] == u:
            user_info.append(d)
    user_info.sort(key=lambda x: x['timestamp'])
    for j in range(len(user_info)):
        rating.append(user_info[j]['rating'])
    return rating

def MSE(y, ypred):
    return sum([(a-b)**2 for (a,b) in zip(y,ypred)]) / len(y)

users = []
for d in dataset:
    if d['user_id'] not in users:
        users.append(d['user_id'])
users_ratings = []
for i in users:
    sub_list = get_all_rating_for_user(i)
    users_ratings.append(sub_list)
ypred = []
y = []
for i in users_ratings:
    if len(i) == 1:
        continue
    y.append(i[-1])
    total = 0
    for j in range(len(i) - 1):
        total += i[j]
    ypred.append(total/(len(i) - 1))

answers['Q1a'] = MSE(y,ypred)
assertFloat(answers['Q1a'])

### 1b
def get_all_rating_for_item(i):
    rating = []
    item_info = []
    for d in dataset:
        if d['book_id'] == i:
            item_info.append(d)
    item_info.sort(key=lambda x: x['timestamp'])
    for j in range(len(item_info)):
        rating.append(item_info[j]['rating'])
    return rating
items = []
for d in dataset:
    if d['book_id'] not in items:
        items.append(d['book_id'])
items_ratings = []
for i in items:
    sub_list = get_all_rating_for_item(i)
    items_ratings.append(sub_list)
ypred = []
y = []
for i in items_ratings:
    if len(i) == 1:
        continue
    y.append(i[-1])
    total = 0
    for j in range(len(i) - 1):
        total += i[j]
    ypred.append(total/(len(i) - 1))
answers['Q1b'] = MSE(y,ypred)
assertFloat(answers['Q1b'])

### 2
def get_y_y_pred(N):
    y = []
    ypred = []
    for i in users_ratings:
        if len(i) == 1:
            continue
        y.append(i[-1])
        if len(i) < N + 1:
            total = 0
            for j in range(len(i) - 1):
                total += i[j]
            ypred.append(total / (len(i) - 1))
        else:
            total = 0
            for j in range(1,N + 1):
                total += i[-1 - j]
            ypred.append(total / N)
    return [y,ypred]

answers['Q2'] = []

for N in [1,2,3]:
    x = get_y_y_pred(N)
    y = x[0]
    ypred = x[1]
    answers['Q2'].append(MSE(y,ypred))
assertFloatList(answers['Q2'], 3)

### 3a
def feature3(N, u): # For a user u and a window size of N
    feat = [1]
    rating = get_all_rating_for_user(u)
    new_rating = []
    if len(rating) < N + 1:
        return -1
    else:
        for i in range(N):
            new_rating.append(rating[-2 - i])
    feat += new_rating
    return feat
answers['Q3a'] = [feature3(2,dataset[0]['user_id']), feature3(3,dataset[0]['user_id'])]
assert len(answers['Q3a']) == 2
assert len(answers['Q3a'][0]) == 3
assert len(answers['Q3a'][1]) == 4

### 3b
answers['Q3b'] = []

for N in [1,2,3]:
    X = []
    Y = []
    for i in users_ratings:
        if len(i) < N + 1:
            continue
        Y.append(i[-1])
        X_item = [1] + i[len(i)-N-1:len(i)-1]
        X.append(X_item)
    X = numpy.matrix(X)
    Y = numpy.matrix(Y)
    result = numpy.linalg.inv(X.T * X) * X.T * Y.T
    result = numpy.matrix.tolist(result)
    X = []
    Y = []
    for i in users_ratings:
        if len(i) < N + 1:
            continue
        Y.append(i[-1])
        X_item = [1] + i[len(i) - N - 1:len(i) - 1]
        X.append(X_item)
    square_sum = 0
    if N == 1:
        theta0 = result[0][0]
        theta1 = result[1][0]
        for i in range(len(Y)):
            square_sum += (theta0 + X[i][1] * theta1 - Y[i]) ** 2
        mse = square_sum / len(Y)
    elif N == 2:
        theta0 = result[0][0]
        theta1 = result[1][0]
        theta2 = result[2][0]
        for i in range(len(Y)):
            square_sum += (theta0 + X[i][1] * theta1 + X[i][2] * theta2 - Y[i]) ** 2
        mse = square_sum / len(Y)
    else:
        theta0 = result[0][0]
        theta1 = result[1][0]
        theta2 = result[2][0]
        theta3 = result[3][0]
        for i in range(len(Y)):
            square_sum += (theta0 + X[i][1] * theta1 + X[i][2] * theta2 + X[i][3] * theta3 - Y[i]) ** 2
        mse = square_sum / len(Y)

    answers['Q3b'].append(mse)

assertFloatList(answers['Q3b'], 3)

### 4a
globalAverage = [d['rating'] for d in dataset]
globalAverage = sum(globalAverage) / len(globalAverage)
def featureMeanValue(N, u): # For a user u and a window size of N
    feat = [1]
    rating = get_all_rating_for_user(u)
    if len(rating) == 1:
        new_rating = []
        for i in range(N):
            new_rating.append(globalAverage)
        return feat + new_rating
    if len(rating) < N + 1:
        new_rating = []
        for i in range(1,len(rating)):
            new_rating.append(rating[-i - 1])
        rating_average = sum(new_rating) / len(new_rating)
        diff = N - len(new_rating)
        for i in range(diff):
            new_rating.append(rating_average)
        return feat + new_rating
    else:
        new_rating = []
        for i in range(N):
            new_rating.append(rating[-2 - i])
        return feat + new_rating

def featureMissingValue(N, u):
    feat = [1]
    rating = get_all_rating_for_user(u)
    if len(rating) < N + 1:
        new_rating = []
        for i in range(1,len(rating)):
            new_rating.append(0)
            new_rating.append(rating[-i - 1])
        diff = int((2 * N - len(new_rating))/2)
        for i in range(diff):
            new_rating.append(1)
            new_rating.append(0)
        return feat + new_rating
    else:
        new_rating = []
        for i in range(N):
            new_rating.append(0)
            new_rating.append(rating[-2 - i])
        return feat + new_rating


answers['Q4a'] = [featureMeanValue(10, dataset[0]['user_id']), featureMissingValue(10, dataset[0]['user_id'])]
assert len(answers['Q4a']) == 2
assert len(answers['Q4a'][0]) == 11
assert len(answers['Q4a'][1]) == 21

### 4b
answers['Q4b'] = []

for featFunc in [featureMeanValue, featureMissingValue]:
    X = []
    Y = []
    lr = linear_model.LinearRegression()
    for u in users:
        Y.append(get_all_rating_for_user(u)[-1])
        X_item = featFunc(10,u)
        X.append(X_item)
    lr.fit(X, Y)
    Y_pre = lr.predict(X)
    mse = sklearn.metrics.mean_squared_error(Y, Y_pre)
    answers['Q4b'].append(mse)
assertFloatList(answers["Q4b"], 2)

### 5
def feature5(sentence):
    feat = [1]
    x1 = len(sentence)
    x2 = 0
    x3 = 0
    for i in sentence:
        if i.isupper():
            x2 += 1
        if i == '!':
            x3 += 1
    return feat + [x1,x2,x3]
y = []
X = []
for d in dataset:
    for spoiler,sentence in d['review_sentences']:
        X.append(feature5(sentence))
        y.append(spoiler)
answers['Q5a'] = X[0]

mod = linear_model.LogisticRegression(C=1.0,class_weight='balanced')
mod.fit(X, y)
predictions = mod.predict(X)
TP = 0
TN = 0
FP = 0
FN = 0
BER = 0
for i in range(len(y)):
    if predictions[i] == 1 and y[i] == 1:
        TP += 1
    elif predictions[i] == 0 and y[i] == 1:
        FN += 1
    elif predictions[i] == 1 and y[i] == 0:
        FP += 1
    else:
        TN += 1
FPR = FP / (FP + TN)
FNR = FN / (TP + FN)
BER = (FPR + FNR) / 2
answers['Q5b'] = [TP, TN, FP, FN, BER]
assert len(answers['Q5a']) == 4
assertFloatList(answers['Q5b'], 5)

### 6
def feature6(review):
    feat = [1]
    sentences = d['review_sentences']
    x = []
    for i in range(5):
        x.append(sentences[i][0])
    return x + feature5(sentences[5][1])



y = []
X = []

for d in dataset:
    sentences = d['review_sentences']
    if len(sentences) < 6: continue
    X.append(feature6(d))
    y.append(sentences[5][0])
mod = linear_model.LogisticRegression(C=1.0,class_weight='balanced')
mod.fit(X, y)
predictions = mod.predict(X)
TP = 0
TN = 0
FP = 0
FN = 0
BER = 0
for i in range(len(y)):
    if predictions[i] == 1 and y[i] == 1:
        TP += 1
    elif predictions[i] == 0 and y[i] == 1:
        FN += 1
    elif predictions[i] == 1 and y[i] == 0:
        FP += 1
    else:
        TN += 1
FPR = FP / (FP + TN)
FNR = FN / (TP + FN)
BER = (FPR + FNR) / 2

answers['Q6a'] = X[0]
answers['Q6b'] = BER
assert len(answers['Q6a']) == 9
assertFloat(answers['Q6b'])

### 7
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
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]
bers = []
C_value = 0.01
for i in range(5):
    mod = linear_model.LogisticRegression(C=C_value,class_weight='balanced')
    mod.fit(Xtrain,ytrain)
    pred_valid = mod.predict(Xvalid)
    berValid = BER(pred_valid,yvalid)
    bers.append(berValid)
    C_value = C_value * 10
ber_best = bers[0]
C_value_index = 0
for i in range(len(bers)):
    if bers[i] < ber_best:
        ber_best = bers[i]
        C_value_index = i
bestC = 10**(-2 + C_value_index)
mod = linear_model.LogisticRegression(C=bestC,class_weight='balanced')
mod.fit(Xtrain,ytrain)
pred_test = mod.predict(Xtest)
ber = BER(pred_test,ytest)

answers['Q7'] = bers + [bestC] + [ber]
assertFloatList(answers['Q7'], 7)

### 8
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom
# 75/25% train/test split
dataTrain = dataset[:15000]
dataTest = dataset[15000:]
# A few utilities

itemAverages = defaultdict(list)
ratingMean = []

for d in dataTrain:
    itemAverages[d['book_id']].append(d['rating'])
    ratingMean.append(d['rating'])

for i in itemAverages:
    itemAverages[i] = sum(itemAverages[i]) / len(itemAverages[i])

ratingMean = sum(ratingMean) / len(ratingMean)
reviewsPerUser = defaultdict(list)
usersPerItem = defaultdict(set)

for d in dataTrain:
    u,i = d['user_id'], d['book_id']
    reviewsPerUser[u].append(d)
    usersPerItem[i].add(u)
# From my HW2 solution, welcome to reuse
def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            return ratingMean
predictions = [predictRating(d['user_id'],d['book_id']) for d in dataTest]
labels = [d['rating'] for d in dataTest]
answers["Q8"] = MSE(predictions, labels)
assertFloat(answers["Q8"])

### 9
a_list = []
b_list = []
c_list = []
for d in dataTest:
    item = d['book_id']
    times = 0
    for t in dataTrain:
        if t['book_id'] == item:
            times += 1
    if times == 0:
        a_list.append(d)
    elif times > 0 and times <= 5:
        b_list.append(d)
    elif times > 5:
        c_list.append(d)
prediction_a = [predictRating(d['user_id'],d['book_id']) for d in a_list]
label_a = [d['rating'] for d in a_list]
mse0 = MSE(prediction_a, label_a)

prediction_b = [predictRating(d['user_id'],d['book_id']) for d in b_list]
label_b = [d['rating'] for d in b_list]
mse1to5 = MSE(prediction_b, label_b)

prediction_c = [predictRating(d['user_id'],d['book_id']) for d in c_list]
label_c = [d['rating'] for d in c_list]
mse5 = MSE(prediction_c, label_c)

answers["Q9"] = [mse0, mse1to5, mse5]
assertFloatList(answers["Q9"], 3)

def newPredictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            if len(reviewsPerUser[user]) != 0:
                total = 0
                for d in reviewsPerUser[user]:
                    total += d['rating']
                rating = total / len(reviewsPerUser[user])
                return rating
            else:
                return ratingMean



new_prediction = [newPredictRating(d['user_id'],d['book_id']) for d in a_list]
label = [d['rating'] for d in a_list]
itsMSE = MSE(new_prediction, label)
improvement = '''
I change the prediction function by adding some judgements to handle the cold-start problem. In the original prediction function,
If the item does not exist in the train dataset, it will return ratingMean as the prediction of the rating of this item. However, this is
not exact. In new prediction function, when item does not exist in the train dataset, it will give the prediction of the rating of this item
based on the rating of the user to all the items which he or she has made a rating in the train dataset. Therefore, we based on the information
of that user to other items to predict the rating of the new test item. This will be much more creditable than just using the ratingMean.
'''

answers["Q10"] = (improvement, itsMSE)
assert type(answers["Q10"][0]) == str
assertFloat(answers["Q10"][1])

f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()