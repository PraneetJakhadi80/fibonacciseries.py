:
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import f1_score,accuracy_score
data = pd.read_csv('mnist_train.csv')
data.head()
data.shape
a = data.iloc[3,1:].values
# Reshape 
a= a.reshape(28,28).astype('uint8')
# Plotting in matplotlib
plt.imshow(a)
data.isnull().sum()
data.dtypes
x = data.drop(['label'],axis=1)
y = data['label']

x.shape
train_x,test_x,train_y,test_y = tts(x,y,test_size=0.2,random_state=96)
train_y.head()

print(train_x.shape)
print(train_y.shape)

index = ['Random Forest Classifier','Logistic Regression','SGD Classifier','KNN-Classifier','Decision Tree Classifier','Naive Bayes']
col = ['Accuracy']
model = pd.DataFrame(index=index,columns=col)
rf = RandomForestClassifier(n_estimators=100)
In [18]:
# fitting data
rf.fit(train_x,train_y)
Out[18]:
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
pred = rf.predict(test_x)

pred

array([3, 7, 2, ..., 9, 0, 7], dtype=int64)


f1 = f1_score(pred,test_y,average='weighted')
print("F1 Score : ",f1)
acc_rf = accuracy_score(pred,test_y)
print("Accuracy : ",acc_rf)
F1 Score :  0.964928336023329
Accuracy :  0.9649166666666666
model.iat[0,0] = acc_rf
model
lr = LogisticRegression(random_state=96)


lr.fit(train_x,train_y)


pred = lr.predict(test_x)

model.iat[1,0] = acc_lr
model
# Calling classifier
sgd = SGDClassifier(random_state=96)

# fitting data
sgd.fit(train_x,train_y)cy
f1 = f1_score(pred,test_y,average='weighted')
print("F1 Score : ",f1)
acc_lr = accuracy_score(pred,test_y)
print("Accuracy : ",acc_lr)
pred = sgd.predict(test_x)
# Checking f1-score and accuracy
f1 = f1_score(pred,test_y,average='weighted')
print("F1 Score : ",f1)
acc_sgd = accuracy_score(pred,test_y)
print("Accuracy : ",acc_sgd

model.iat[2,0] = acc_sgd
model
knn = KNeighborsClassifier(n_neighbors=5)
In [32]:
# fitting the data
knn.fit(train_x,train_y)
Out[32]:
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')

pred = knn.predict(test_x)

acc_knn = accuracy_score(pred,test_y)
print("Accuracy : ",acc_knn)


f1 = f1_score(pred,test_y,average='weighted')
print("F1 Score : ",f1)


model.iat[3,0] = acc_knn
model
def Elbow(K):
    error=[]
    for i in K:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_x,train_y)
        tmp = knn.predict(test_x)
        tmp =f1_score(pred,test_y,average='weighted')
        error.append(tmp)
    return error
k=range(1,50)
dec = DecisionTreeClassifier(random_state=96)

# fitting data
dec.fit(train_x,train_y)

# Checking f1-score and accuracy
f1 = f1_score(pred,test_y,average='weighted')
print("F1 Score : ",f1)
acc_dec = accuracy_score(pred,test_y)
print("Accuracy : ",acc_dec)

model.iat[4,0] = acc_dec
model

gnb = GaussianNB()

# fitting data
gnb.fit(train_x,train_y)

pred = gnb.predict(test_x)
# Checking f1-score and accuracy
f1 = f1_score(pred,test_y,average='weighted')
print("F1 Score : ",f1)
acc_gnb = accuracy_score(pred,test_y)
print("Accuracy : ",acc_gnb)
mnb = MultinomialNB()

# fitting data
mnb.fit(train_x,train_y)



pred = mnb.predict(test_x)
# Checking f1-score and accuracy
f1 = f1_score(pred,test_y,average='weighted')
print("F1 Score : ",f1)
acc_mnb = accuracy_score(pred,test_y)
print("Accuracy : ",acc_mnb)

bnb = BernoulliNB()

bnb.fit(train_x,train_y)
pred = bnb.predict(test_x)
# Checking f1-score and accuracy
f1 = f1_score(pred,test_y,average='weighted')
print("F1 Score : ",f1)
acc_bnb = accuracy_score(pred,test_y)
print("Accuracy : ",acc_bnb)


model.iat[5,0] = acc_bnb
model
    
