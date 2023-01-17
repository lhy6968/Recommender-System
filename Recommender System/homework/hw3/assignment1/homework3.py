import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model

import warnings
warnings.filterwarnings("ignore")

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

answers = {}

# Some data structures that will be useful
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

ratingsTrain = allRatings[:190000]
print(ratingsTrain[0])
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))


# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break

### Question 1
users_books_dict = {}
all_books = []
for u,b,r in allRatings:
    if u not in users_books_dict:
        users_books_dict[u] = []
        users_books_dict[u].append(b)
    else:
        users_books_dict[u].append(b)
    all_books.append(b)
all_books = list(set(all_books))
users_negative_books_dict = {}

negative_valid_list = []
false_num_0 = 0
for u,b,r in ratingsValid:
    if b not in return1:
        false_num_0 += 1
    unlooked_book_list = list(set(all_books).difference(set(users_books_dict[u])))
    new_b = random.choice(unlooked_book_list)
    negative_valid_list.append([u,new_b])

false_num = 0
for i in negative_valid_list:
    if i[1] in return1:
        false_num += 1

accuracy = 1 - (false_num  + false_num_0) / (len(ratingsValid) + len(negative_valid_list))
answers['Q1'] = accuracy

assertFloat(answers['Q1'])



### Question 2
threshold = 0.6
return2 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return2.add(i)
    if count > totalRead * threshold: break

false_num_0 = 0
for u,b,r in ratingsValid:
    if b not in return2:
        false_num_0 += 1

false_num = 0
for i in negative_valid_list:
    if i[1] in return2:
        false_num += 1
accuracy = 1 - (false_num_0 + false_num) / (len(ratingsValid) + len(negative_valid_list))
answers['Q2'] = [threshold, accuracy]

assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])

### Question 3/4
usersPerItem = defaultdict(set)
for u,b,r in ratingsTrain:
    usersPerItem[b].add(u)

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom

def mostSimilar(i,u,N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i:
            continue
        elif u in usersPerItem[i2]:
            sim = Jaccard(users,usersPerItem[i2])
            similarities.append(sim)
    similarities.sort(reverse=True)
    return similarities[:N]

similar = []
for u,b,r in ratingsValid:
    sub_similar = mostSimilar(b,u,1)
    if sub_similar == []:
        similar.append(0)
    else:
        similar.append(sub_similar[0])
similar_2 = []
for i in negative_valid_list:
    sub_similar = mostSimilar(i[1],i[0],1)
    if sub_similar == []:
        similar_2.append(0)
    else:
        similar_2.append(sub_similar[0])

false_num_0 = 0
false_num = 0
for i in range(len(similar)):
    if similar[i] <= 0.0001:
        false_num_0 += 1
    if similar_2[i] > 0.0001:
        false_num += 1


accuracy = 1 - (false_num_0 + false_num) / (len(ratingsValid) + len(negative_valid_list))
answers['Q3'] = accuracy
assertFloat(answers['Q3'])


false_num_0 = 0
false_num = 0
for i in range(len(similar)):
    if similar[i] <= 0.0001 and ratingsValid[i][1] not in return2:
        false_num_0 += 1
    elif similar_2[i] > 0.0001 and ratingsValid[i][1] in return2:
        continue
    else:
        if similar[i] <= 0.0001:
            false_num += 1
        else:
            continue

for i in range(len(similar_2)):
    if similar_2[i] > 0.0001 and negative_valid_list[i][1] in return2:
        false_num += 1
    elif similar_2[i] <= 0.0001 and negative_valid_list[i][1] not in return2:
        continue
    else:
        if similar_2[i] > 0.0001:
            false_num += 1
        else:
            continue



accuracy = 1 - (false_num_0 + false_num) / (len(ratingsValid) + len(negative_valid_list))


answers['Q4'] = accuracy
assertFloat(answers['Q4'])


predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    similar = mostSimilar(b,u,1)
    if similar == []:
        similar = 0
    else:
        similar = similar[0]
    if b in return2 and similar > 0.0001:
        predictions.write(u + ',' + b + ",1\n")
    elif b not in return2 and similar <= 0.0001:
        predictions.write(u + ',' + b + ",0\n")
    else:
        if similar > 0.0001:
            predictions.write(u + ',' + b + ",1\n")
        else:
            predictions.write(u + ',' + b + ",0\n")
predictions.close()

answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"
assert type(answers['Q5']) == str


### Question 9
ratingsum = 0
for i in range(len(ratingsTrain)):
    ratingsum += ratingsTrain[i][2]
ratingMean = ratingsum / len(ratingsTrain)
alpha = ratingMean
userBiases = defaultdict(float)
itemBiases = defaultdict(float)
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    reviewsPerUser[u].append(b)
    reviewsPerItem[b].append(u)
N = len(ratingsTrain)
nUsers = len(reviewsPerUser)
nItems = len(reviewsPerItem)
users = list(reviewsPerUser.keys())
items = list(reviewsPerItem.keys())
def prediction(user, item):
    if user not in userBiases:
        b = 0
        userBiases[user] = 0.0
    else:
        b = userBiases[user]
    if item not in itemBiases:
        c = 0
        userBiases[user] = 0.0
    else:
        c = itemBiases[item]
    return alpha +  + b + c
def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))
mse_list = []
def cost(theta, labels, lamb):
    global mse_list
    unpack(theta)
    predictions = [prediction(i[0],i[1]) for i in ratingsValid]
    cost = MSE(predictions, labels)
    mse_list.append(cost)
    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in itemBiases:
        cost += lamb*itemBiases[i]**2
    return cost

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(ratingsTrain)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    for u,b,r in ratingsTrain:
        pred = prediction(u, b)
        diff = pred - r
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[b] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in itemBiases:
        dItemBiases[i] += 2*lamb*itemBiases[i]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
    return numpy.array(dtheta)

labels = [i[2] for i in ratingsValid]
scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                             derivative, args = (labels, 1))


validMSE = mse_list[-1]

answers['Q9'] = validMSE

assertFloat(answers['Q9'])

### Question 10
#answers['Q10'] = [maxUser, minUser, maxBeta, minBeta]
maxUser = ''
minUser = ''
maxBeta = float(0)
minBeta = float(0)
for i in userBiases:
    if maxUser == '':
        maxUser = i
        minUser = i
        maxBeta = float(userBiases[i])
        minBeta = float(userBiases[i])
    else:
        if userBiases[i] > maxBeta:
            maxBeta = float(userBiases[i])
            maxUser = i
        if userBiases[i] < minBeta:
            minBeta = float(userBiases[i])
            minUser = i
answers['Q10'] = [maxUser, minUser, maxBeta, minBeta]
#print([maxUser, minUser, maxBeta, minBeta])

assert [type(x) for x in answers['Q10']] == [str, str, float, float]

lamb = 0.01
labels = [i[2] for i in ratingsValid]
scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                             derivative, args = (labels, lamb))



validMSE = mse_list[-1]
#print(validMSE)
answers['Q11'] = (lamb, validMSE)

assertFloat(answers['Q11'][0])
assertFloat(answers['Q11'][1])

f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()






