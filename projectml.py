import sys
print('python : {}'.format(sys.version))
import scipy
print('scipy : {}'.format(sys.version))
import numpy
print('numpy : {}'.format(sys.version))
import matplotlib
print('matplotlib : {}'.format(sys.version))
import sklearn
print('sklearn : {}'.format(sys.version))
import pandas
print('pandas : {}'.format(sys.version))

import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master.iris.csv'
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=read_csv(url,names=names)
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(datsaset.groupby('class').size())
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
pyplot.show()
dataset.hist()
pyplot.show()
scatter_matrix(dataset)
pyplot.show()
array=dataset.values
x=array[:,0:4]
y=array[:,4]
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=1)
model=[]
model.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
model.append(('LDA',LinearDiscriminantAnalysis()))
model.append(('KNN',KNeighborsClassifier()))
model.append(('NB',GaussianNB()))
model.append(('SVM',SVC(gamma='auto')))
result=[]
names=[]
for name,model in models:
  kfold=StratifiedKFold(n_splits=10, random_state=1)
  cv_results=cross_val_score(model, x_train, y_train, groups=None, scoring='accuracy', cv=kfold )
  result.append(cv_results)
  names.append(name)
  print('%s: %f (%f) ' % (name,cv_results.mean(),cv_results.std()))
  pyplot.boxplot(results,labels=names)
pyplot.title('algo  comp ')
pyplot.show()
model=SVC(gamma='auto')
model.fit(x_train,y_train)
predictions=model_selection(x_val)
print(accuracy_score(x_val,predictions))
print(confusion_matrix(x_val,predictions))
print(classification_report(x_val,predictions))
