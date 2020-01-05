

import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

#Load Data From CSV Fil
df = pd.read_csv('loan_train.csv')
print(df.head())

print(df.shape)

#Convert to date time object
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
print(df.head())

#Data visualization and pre-processing
df['loan_status'].value_counts()

import seaborn as sns
#principal
bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

#age
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

#Pre-processing: Feature selection/extraction
#Lets look at the day of the week people get the loan
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

#
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
print(df.head())

df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

#Convert Categorical features to numerical values
# df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
print(df.groupby(['Gender'])['loan_status'].value_counts(normalize=True))

#Lets convert male to 0 and female to 1:
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()

df.groupby(['education'])['loan_status'].value_counts(normalize=True)

df[['Principal','terms','age','Gender','education']].head()
Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

X = Feature
print(X[0:5])

y = df['loan_status'].values
print(y[0:5])

#Normalize Data
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

#Classification
#---------------------------k-nearest
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
k = 7
# train model and predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
neigh
#predicting
yhat = neigh.predict(X_test)
yhat[0:5]
#accuracy evaluation
from  sklearn import metrics
print("Train set accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set accuracy: ",metrics.accuracy_score(y_test,yhat))

#--------------------------------------decision tree
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

#modeling
from sklearn.tree import DecisionTreeClassifier
modelTree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
modelTree.fit(X_trainset,y_trainset)
#prediction
predTree = modelTree.predict(X_testset)
print(predTree[0:5])
print(y_testset[0:5])

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy:", metrics.accuracy_score(y_testset,predTree))


#--------------------------------------suport vector machine
#train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# kernel function (default - radial basis function)
from sklearn import svm
# clf = svm.SVC(gamma='auto', kernel='linear')
clf = svm.SVC(gamma='auto', kernel='rbf')
clf.fit(X_train, y_train)

#predict
yhat = clf.predict(X_test)
yhat [0:5]


#-----------------------------------Logistic Regression
#train/test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

#modeling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

#predict test set
yhat = LR.predict(X_test)

#__predict_proba__
yhat_prob = LR.predict_proba(X_test)