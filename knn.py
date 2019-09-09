import numpy as np
import scipy.io as sio
import operator
from sklearn.metrics import confusion_matrix

#####################################
######### Loading Datasets ##########
#####################################
ch = input("Press 1 to check on USPS-dataset\nPress 2 to check on MNIST-dataset\n")
if ch=='1':
    data = sio.loadmat('./USPS.mat')
else:
    data = sio.loadmat('./MNIST.mat')
X_train = data['train_data'].T
y_train = data['train_lbl']
X_test = data['test_data'].T
y_test = data['test_lbl']

#######################################
######### K-Nearest Neighbor ##########
#######################################
def knn(k,test):
    global X_train, y_train, X_test, y_test

    distList = []
    for j in range(len(X_train)):
        dist = np.linalg.norm(X_train[j]-test)
        distTup = np.asscalar(y_train[j]), dist
        distList.append(distTup)
    
    sortedTup = sorted(distList, key=lambda tup:tup[1])

    neighbors = []
    cnt = 0
    for (a,b) in sortedTup:
        neighbors.append([a,b])
        cnt+=1
        if cnt==k:
            break
    return neighbors

def prediction(k):
    global X_train, y_train, X_test, y_test
    y_pred = []
    for i in range(len(X_test)):
        test = X_test[i]
        neighbors = knn(k,test)
        votes = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
        for j in neighbors:
            votes[j[0]] += 1
        pred = max(votes.items(), key=operator.itemgetter(1))[0]
        y_pred.append(pred)
        print("Done test instance: ",i)
        # if i==5:
        #     break
    return np.asarray(y_pred)


k = int(input("Enter k "))
#k = 5
y_pred = prediction(k)
cm = confusion_matrix(y_test, y_pred)
print(cm)
