#!/usr/bin/env python
# coding: utf-8

# # FAKE NEWS DETECTION

# In[1]:


# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Loading Dataset

df_fake=pd.read_csv('Fake.csv')
df_true=pd.read_csv('True.csv')


# In[3]:


df_fake.head(10)


# In[4]:


df_true.head(10)


# In[5]:


df_fake.shape, df_true.shape


# In[6]:


df_fake.columns, df_true.columns


# In[7]:


df_fake.dtypes, df_true.dtypes


# In[8]:


df_fake['class']=0
df_true['class']=1


# In[9]:


df_fake.shape, df_true.shape


# ## Taking last 10 values for manual testing from both Datasets

# In[10]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23480, 23470, -1):
    df_fake.drop([i], axis=0, inplace=True)  # Removing last 10 rows of both the original dataset


# In[11]:


df_true_manual_testing = df_true.tail(10)    
for i in range(21416, 21406, -1):
    df_true.drop([i], axis=0, inplace=True)


# In[12]:


df_fake.shape, df_true.shape


# In[13]:


## merging these two datasets in single dataframe

df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("manual_testing.csv ")


# In[14]:


## Merging main two datasets 
 
df_merge = pd.concat([df_fake,df_true],axis=0)
df_merge.head(10)      # displaying first 10 values after above operation


# In[15]:


df_merge.columns


# # Data Visualisations

# In[16]:


df_merge.groupby(['subject'])['text'].count()  # we have grouped subject


# In[17]:


df_merge.groupby(['subject'])['text'].count().plot(kind="bar")
plt.title("Articles per subject",size=20)
plt.xlabel("Category",size=15)
plt.ylabel("Article count",size=15)
plt.show()


# In[18]:


df_merge.groupby(['class'])['text'].count()


# In[19]:


print("0 = Fake news\n1 = True news")
df_merge.groupby(['class'])['text'].count().plot(kind="pie")
plt.title("Fake news and True News",size=15)
plt.show()`


# In[20]:


## removing 3 uncessary columns from the dataset


# In[21]:


df = df_merge.drop(['title','date','subject'], axis=1)


# In[22]:


df.head(10)


# In[23]:


df.isnull().sum() # to check missing values


# In[24]:


df= df.sample(frac=1)  # random shuffling (1- true and 0-fake)


# In[25]:


df.head(10)


# In[26]:


df.reset_index(inplace=True)
df.drop(['index'], axis=1, inplace= True)


# In[27]:


df.head(10)


# In[28]:


## creating a filtering function to remove unwanted data from text

def filtering(data):
    text=data.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\s+|www\.S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    return text


# In[29]:


df["text"]= df["text"].apply(filtering)
df.head(10)


# # Splitting Dataset into Training and Testing

# In[30]:


# Creating Dependant and independant variables

x=df['text']
y=df['class']


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# ## Vectorizing the text 

# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[34]:


vectorization = TfidfVectorizer()
#IDF returns numerical statics that how the word is important to the document

xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)


# # Classifaction using various classifers

# ## (i) Logistic Regression 

# In[35]:


from sklearn.linear_model import LogisticRegression


# In[36]:


LR = LogisticRegression()
LR.fit(xv_train,y_train)  # training or fitting the model to the training set


# In[37]:


LR.score(xv_test,y_test)  # computes the accuracy score


# In[38]:


pred_LR= LR.predict(xv_test) # Predict using the linear model


# In[39]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_LR))


# In[40]:


## Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,pred_LR)  
print(cm)


# In[41]:


sns.heatmap(cm,cmap="BuPu",annot=True)


# ## (ii) Decision Tree Classifier

# In[42]:


from sklearn.tree import DecisionTreeClassifier


# In[43]:


DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[44]:


DT.score(xv_test, y_test)


# In[45]:


pred_DT = DT.predict(xv_test)


# In[46]:


print(classification_report(y_test,pred_DT))


# In[47]:


cm= confusion_matrix(y_test,pred_DT)  
print(cm)


# In[48]:


sns.heatmap(cm,cmap="PiYG",annot=True)


# ## (iii) Random Forest Classifier 

# In[49]:


from sklearn.ensemble import RandomForestClassifier


# In[50]:


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train,y_train)


# In[51]:


RFC.score(xv_test, y_test)


# In[52]:


pred_RFC = RFC.predict(xv_test)


# In[53]:


print(classification_report(y_test, pred_RFC))


# In[54]:


cm= confusion_matrix(y_test,pred_RFC)  
print(cm)


# In[55]:


sns.heatmap(cm,cmap="Blues",annot=True)


# ## (iv) Gradient Boosting Classifier

# In[56]:


from sklearn.ensemble import GradientBoostingClassifier


# In[57]:


GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train,y_train)


# In[60]:


GBC.score(xv_test,y_test)


# In[61]:


pred_GBC = GBC.predict(xv_test)


# In[62]:


print(classification_report(y_test,pred_GBC))


# In[63]:


cm= confusion_matrix(y_test,pred_GBC)  
print(cm)


# In[64]:


import seaborn as sns
sns.heatmap(cm,cmap="Greens",annot=True)


# ## (v) Naive Bayes

# In[63]:


from sklearn.naive_bayes import BernoulliNB


# In[66]:


NB = BernoulliNB()
NB.fit(xv_train,y_train)


# In[67]:


NB.score(xv_test,y_test)


# In[68]:


pred_NB = NB.predict(xv_test)


# In[69]:


print(classification_report(y_test,pred_NB))


# In[71]:


cm= confusion_matrix(y_test,pred_NB)
cm


# In[72]:


import seaborn as sns
sns.heatmap(cm,cmap="copper",annot=True)


# ## (vi) Support Vector Machine

# In[73]:


from sklearn import svm


# In[74]:


#Create a svm Classifier
SV = svm.SVC(kernel='linear')

#Train the model using the training sets
SV.fit(xv_train, y_train)


# In[75]:


SV.score(xv_test, y_test)


# In[76]:


pred_SV = clf.predict(xv_test)


# In[77]:


print(classification_report(y_test, pred_SV))


# In[78]:


cm= confusion_matrix(y_test,pred_SV)  
print(cm)


# In[80]:


sns.heatmap(cm,cmap="spring",annot=True)


# # K Nearest Neighbor

# In[81]:


from sklearn.neighbors import KNeighborsClassifier


# In[82]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xv_train, y_train)


# In[83]:


knn.score(xv_test, y_test)


# In[84]:


pred_knn = knn.predict(xv_test)


# In[85]:


print(classification_report(y_test, pred_knn))


# In[87]:


cm= confusion_matrix(y_test,pred_knn)  
print(cm)


# In[91]:


sns.heatmap(cm,cmap="summer_r",annot=True)

