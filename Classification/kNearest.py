import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing

df = pd.read_csv('teleCust1000t.csv')
df.head()

df['custcat'].value_counts()
df.hist(column='income',bins=50)

#normalize data
X = df[['region','tenure','age','marital','address','income','ed','employ','retire','gender','reside']].values
X[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float));

y = df['custcat'].values
y[0:5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

k = 6
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

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = []

#calculate the accuracy of knn from k = 1 to k = 10
for n in range(1,Ks):
    # train model and predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuaracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

# f1 score
from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted'))

print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)