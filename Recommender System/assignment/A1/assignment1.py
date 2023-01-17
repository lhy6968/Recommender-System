import gzip
import random
from collections import defaultdict
import scipy.optimize
import numpy
import warnings
import surprise
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
warnings.filterwarnings("ignore")

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

# Question 1 Read Prediction
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append([l[0]]+[l[1]]+[int(l[2])])

'''
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
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
  if count > totalRead * 0.74: break

predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
    #header
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    sub_similar = mostSimilar(b,u,1)
    if b in return1:
        predictions.write(u + ',' + b + ",1\n")
    else:
        if sub_similar == []:
            predictions.write(u + ',' + b + ",0\n")
        elif sub_similar[0] <= 0.05:
            predictions.write(u + ',' + b + ",0\n")
        else:
            predictions.write(u + ',' + b + ",1\n")
predictions.close()
'''

# Question 2 Rating Prediction
with open("train", 'w', newline='') as f:
    for rating in allRatings:
        total_element = [str(element) for element in rating]
        f.write(' '.join(total_element) + '\n')


reader = Reader(line_format='user item rating', sep=' ')
data = Dataset.load_from_file("train", reader=reader)
trainset, testset = train_test_split(data, test_size=.05)
model=surprise.SVD(n_factors=2,n_epochs=120,reg_pu=1e-4,reg_qi=1e-4,lr_all=0.001)
model.fit(trainset)
predictions = model.test(testset)
sse = 0
for p in predictions:
    sse += (p.r_ui - p.est)**2

predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, b = l.strip().split(',')
    predictions.write(u + ',' + b + ',' + str(model.predict(u, b).est) + '\n')
predictions.close()