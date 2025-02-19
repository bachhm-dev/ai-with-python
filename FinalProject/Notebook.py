#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[7]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[14]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[15]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[16]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[17]:


X = Feature
X[0:5]


# What are our lables?

# In[18]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[19]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[20]:


from sklearn.model_selection import train_test_split
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train_knn.shape, y_train_knn.shape)
print ('Test set:', X_test_knn.shape, y_test_knn.shape)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
k = 7
# train model and predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train_knn, y_train_knn)
neigh
#predicting
yhat_knn = neigh.predict(X_test_knn)
yhat_knn[0:5]


# In[22]:


#accuracy evaluation
from  sklearn import metrics
print("Train set accuracy: ", metrics.accuracy_score(y_train_knn, neigh.predict(X_train_knn)))
print("Test set accuracy: ",metrics.accuracy_score(y_test_knn, yhat_knn))


# # Decision Tree

# In[23]:


from sklearn.model_selection import train_test_split
X_trainset_decisiontree, X_testset_decisiontree, y_trainset_decisiontree, y_testset_decisiontree = train_test_split(X, y, test_size=0.3, random_state=3)


# In[24]:


#modeling
from sklearn.tree import DecisionTreeClassifier
modelTree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
modelTree.fit(X_trainset_decisiontree,y_trainset_decisiontree)
#prediction
predTree = modelTree.predict(X_testset_decisiontree)
print(predTree[0:5])
print(y_testset_decisiontree[0:5])


# In[25]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy:", metrics.accuracy_score(y_testset_decisiontree,predTree))


# # Support Vector Machine

# In[26]:


#train model
X_train_vector, X_test_vector, y_train_vector, y_test_vector = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train_vector.shape,  y_train_vector.shape)
print ('Test set:', X_test_vector.shape,  y_test_vector.shape)


# In[27]:


# kernel function (default - radial basis function)
from sklearn import svm
# clf = svm.SVC(gamma='auto', kernel='linear')
clf = svm.SVC(gamma='auto', kernel='rbf')
clf.fit(X_train_vector, y_train_vector)


# In[28]:


#predict
yhat_vector = clf.predict(X_test_vector)
yhat_vector [0:5]


# # Logistic Regression

# In[29]:


#train/test dataset
from sklearn.model_selection import train_test_split
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train_logistic.shape,  y_train_logistic.shape)
print('Test set:', X_test_logistic.shape,  y_test_logistic.shape)


# In[30]:


#modeling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train_logistic, y_train_logistic)
# LR = LogisticRegression(C=0.01, solver='sag').fit(X_train_logistic, y_train_logistic)


# In[31]:


#predict test set
yhat_logistic = LR.predict(X_test_logistic)

#__predict_proba__
yhat_prob_logistic = LR.predict_proba(X_test_logistic)


# # Model Evaluation using Test set

# In[32]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[33]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[34]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[35]:


y_test_evaluation = test_df['loan_status'].values


# In[36]:


yhat_knn = yhat_knn[:54]
f1_score_knn = f1_score(y_test_evaluation, yhat_knn, average='weighted')
f1_score_knn

jaccard_score_knn = jaccard_similarity_score(y_test_evaluation, yhat_knn)
jaccard_score_knn


# In[37]:


predTree = predTree[:54]
f1_score_tree = f1_score(y_test_evaluation, predTree, average='weighted')
f1_score_tree

jaccard_score_tree = jaccard_similarity_score(y_test_evaluation, predTree)
jaccard_score_tree


# In[38]:


yhat_vector = yhat_vector[:54]
f1_score_vector = f1_score(y_test_evaluation, yhat_vector, average='weighted')
f1_score_vector

jaccard_score_vector = jaccard_similarity_score(y_test_evaluation, yhat_vector)
jaccard_score_vector


# In[39]:


yhat_logistic = yhat_logistic[:54]
f1_score_logistic = f1_score(y_test_evaluation, yhat_logistic, average='weighted')
f1_score_logistic

jaccard_score_logistic = jaccard_similarity_score(y_test_evaluation, yhat_logistic)
jaccard_score_logistic


# In[40]:


yhat_prob_logistic = yhat_prob_logistic[:54]
from sklearn.metrics import log_loss
log_loss_logistic = log_loss(y_test_evaluation, yhat_prob_logistic)
log_loss_logistic


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |

# In[41]:


Reports = {'Algorithm': ['KNN','Decision Tree','SVM','LogisticRegression'],
          'Jaccard':[jaccard_score_knn,jaccard_score_tree,jaccard_score_vector,jaccard_score_logistic],
          'F1-score':[f1_score_knn,f1_score_tree,f1_score_vector,f1_score_logistic],
          'Logloss':['NA','NA','NA',log_loss_logistic]}

dfReport = pd.DataFrame(Reports, columns = ['Algorithm','Jaccard','F1-score','Logloss'])
dfReport


# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
