import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import itertools
import pdb

colName = ['lettr','x-box','ybox','width','high','onpix','x-bar','y-bar','x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx']
data = pd.read_csv('LRD.data',header = None, names = colName)

data.head()

data.shape

pd.isnull(data).sum()

plt.subplots(figsize=(8,4))
sns.barplot(x = data['lettr'].value_counts().index, y = data['lettr'].value_counts().values)
plt.xlabel('Classes')
plt.ylabel('Frequency')

def Preprocessing(df):
    labels_train = pd.DataFrame()
    features_train = pd.DataFrame()
    labels_test = pd.DataFrame()
    features_test = pd.DataFrame()
    #Seperate features and labels of each class
    for i in range(26):
        labels_class = df.lettr[df['lettr'] == chr(ord('A') + i)]
        features_class = df[df['lettr'] == chr(ord('A') + i)].drop(['lettr'], axis = 1)
        index = np.random.permutation(len(labels_class)) #Random Indexes to divide the dataset into train and test
        labels_train = pd.concat([labels_train, labels_class.iloc[index[0:650:1]]])
        features_train = pd.concat([features_train, features_class.iloc[index[0:650:1]]])
        labels_test = pd.concat([labels_test, labels_class.iloc[index[650:len(labels_class)]]])
        features_test = pd.concat([features_test, features_class.iloc[index[650:len(labels_class)]]])
    #Shuffle the training data
    shuff = np.random.permutation(len(labels_train))
    labels_train = labels_train.iloc[shuff]
    features_train = features_train.iloc[shuff]
    #Shuffle the test data
    shuff = np.random.permutation(len(labels_test))
    labels_test = labels_test.iloc[shuff]
    features_test = features_test.iloc[shuff]

    return labels_train, features_train, labels_test, features_test

labels_train, features_train, labels_test, features_test = Preprocessing(data)
labels_train.columns = ['lettr']
labels_test.columns = ['lettr']

#Naive Bayes Models

#Find the prior probability of each class
prior = np.zeros((26,1))

for i in range(26):
    prior[i] = (labels_train == chr(ord('A') + i)).sum()/len(labels_train)

#Seperate the features of each class and find the probability matrix(each feature)
likelihood = np.zeros((26,16))

for i in range(26):
    tidx = labels_train == chr(ord('A') + i)
    temp = features_train[tidx.values]
    likelihood[i,:] = (1 + np.sum(temp,axis = 0))/np.sum(temp).sum()
    
#Classify using Multinomial Naivee Bayes
prob = np.zeros((26,1))
pred = np.ndarray(shape=(len(labels_test),1), dtype=object)

for i in range(len(labels_test)):
    for j in range(26):
        prob[j] = np.log(prior[j]) + np.sum(np.log(likelihood[j,:])*features_test.iloc[i])
    maxIndex = np.argmax(prob)
    pred[i] = chr(ord('A') + maxIndex)
    
acc = np.array(np.sum(labels_test == pred)/len(labels_test))
print('Multinomial Naive Bayes:',np.around(acc[0]*100,decimals=3))

cnf = confusion_matrix(labels_test, pred)cnf

#print(classification_report(labels_test, pred, target_names=data.lettr))

#Implement Gaussian Naive Bayes
Meu = np.zeros((26,16)) #Calculate the mean for each class and feature
sigma = np.zeros((26,16)) #Calculate the std for each class and feature

for i in range(26):
    tidx = labels_train == chr(ord('A') + i)
    temp = features_train[tidx.values]
    Meu[i,:] = temp.mean()
    sigma[i,:] = temp.std()

check = np.zeros((26,1))
#Now classify the test data
for i in range(len(labels_test)):
    for j in range(26):
        check[j] = prior[j] * np.product(np.exp(-0.5*(np.square(features_test.iloc[i] - Meu[j,:])/np.square(sigma[j,:])))/(np.sqrt(2*np.pi*sigma[j,:])))
    maxIndex = np.argmax(check)
    pred[i] = chr(ord('A') + maxIndex)


acc = np.array(np.sum(labels_test == pred)/len(labels_test))
print('Gaussian Naive Bayes:',np.around(acc[0]*100,decimals=3))

cnf = confusion_matrix(labels_test, pred)cnf

#print(classification_report(labels_test, pred, target_names=data.lettr))

#Implement Multinomial multivariate Naive Bayes with smoothing
lh_mul = np.zeros((26,16,16)) #Likelihood of each feature of each class

for i in range(26):
    tidx = labels_train == chr(ord('A') + i)
    temp = features_train[tidx.values]
    shp = temp.shape
    for j in range(16):
        for k in range(16):
            indx = temp.iloc[:,j] == k
            lh_mul[i,j,k] = (1 + indx.sum())/(shp[0]+shp[1])

check = np.zeros((26,1))

for i in range(len(labels_test)):
    for j in range(26):
        temp = 0
        for k in range(16):
            temp = np.log(lh_mul[j,k,features_test.iloc[i,k]])*features_test.iloc[i,k]
            if np.isnan(temp):
                temp = 0
            check[j] = check[j] + temp
        check[j] = np.log(prior[j]) + check[j]
    maxIndex = np.argmax(check)
    pred[i] = chr(ord('A') + maxIndex)
    check[:] = 0 

#Implement Multinomial multivariate Naive Bayes with smoothing
lh_mul = np.zeros((26,16,16)) #Likelihood of each feature of each class

for i in range(26):
    tidx = labels_train == chr(ord('A') + i)
    temp = features_train[tidx.values]
    shp = temp.shape
    for j in range(16):
        for k in range(16):
            indx = temp.iloc[:,j] == k
            lh_mul[i,j,k] = (1 + indx.sum())/(shp[0]+shp[1])

check = np.zeros((26,1))

for i in range(len(labels_test)):
    for j in range(26):
        temp = 0
        for k in range(16):
            temp = np.log(lh_mul[j,k,features_test.iloc[i,k]])*features_test.iloc[i,k]
            if np.isnan(temp):
                temp = 0
            check[j] = check[j] + temp
        check[j] = np.log(prior[j]) + check[j]
    maxIndex = np.argmax(check)
    pred[i] = chr(ord('A') + maxIndex)
    check[:] = 0 


# In[18]:


acc = np.array(np.sum(labels_test == pred)/len(labels_test))
print('Multinomial Multivariate Naive Bayes:',np.around(acc[0]*100,decimals=3))


# In[19]:


cnf = confusion_matrix(labels_test, pred)cnf


# In[20]:


#print(classification_report(labels_test, pred, target_names=data.lettr))


# Among the 3 different versions of Naive Bayes, Multinomial Multivariate Naive Bayes performs the best with an accuracy of 69-72% with relatively good precision and recall on average. Lets compare these methods to logistic regression and neural networks

# In[21]:


#Normalize the features matrices for Neural Networks and logistic regression
features_train = features_train/15
features_test = features_test/15


# In[22]:


def softmax(ar):
    exps = np.exp(ar)
    denom = sum(exps)
    softmax = exps/denom
    return softmax


# In[28]:


#Implementation of Logistic regression using neural networks with SGD + momentum

#Initalize weight and bias using Xaviers Initialization
#Source:https://mnsgrg.com/2017/12/21/xavier-initialization/
b = -np.sqrt(6/(16+26)) + 2*np.sqrt(6/(16+26))*np.random.uniform(size = (26,1))
w = -np.sqrt(6/(16+26)) + 2*np.sqrt(6/(16+26))*np.random.uniform(size = (26,16))

#Initialize Variables
epoch = 50
lr = 0.05
alp = 0.35
lbl = np.zeros((26,1))
err = np.zeros((epoch,1))
acc = np.zeros((epoch,1))
db = np.zeros((26,1))   #Gradient for bias
dw = np.zeros((26,16))  #Gradient for weight

#Temp variables to save best parameters
min_err = np.Inf
max_acc = 0
bpred = np.zeros((len(labels_test),1))

for i in range(epoch):
    shuff = np.random.permutation(len(labels_train))
    #Pass over an epoch
    for j in range(len(labels_train)):
        cls = labels_train.iloc[shuff[j]]
        lbl[ord(cls[0]) - 65] = 1    #One hot encoding

        #Forward Pass
        f = np.reshape(np.array(features_train.iloc[shuff[j]]),(16,1)) #Feature needs to be reshaped due to numpy default behaviour
        tot = np.matmul(w,f) + b
        o = softmax(tot)

        #Backward Pass
        db = alp*db + lr*(o-lbl)
        dw = alp*dw + lr*np.matmul((o-lbl),np.transpose(f))
        b = b - db
        w = w - dw

        lbl[:] = 0
    
    #Check performance on test set
    for k in range(len(labels_test)):
        cls = labels_test.iloc[k]
        lbl[ord(cls[0]) - 65] = 1
        f = np.reshape(np.array(features_test.iloc[k]),(16,1))
        tot = np.matmul(w,f) + b
        o = softmax(tot)

        maxIdx = np.argmax(o)
        pred[k] = chr(ord('A') + maxIdx)

        if pred[k] == cls[0]:
            acc[i] = acc[i] + 1

        entropy = -lbl*np.log(o)
        entropy = np.nan_to_num(entropy)
        err[i] = err[i] + np.sum(entropy)
        lbl[:] = 0
    
    #print('Epoch:', i, 'Accuracy:', acc[i]/len(labels_test)*100, 'Cross Entropy Error:', err[i]/len(labels_test))
    #Minimum Error 
    if (err[i] < min_err):
        min_err = err[i]
        bpred = pred

    #Set gradient and labels back to zero
    lbl[:] = 0
    db[:] = 0
    dw[:] = 0


# In[24]:


print('Neural Network with Logistic Regression:',np.around(np.max(acc)/len(labels_test)*100,decimals=3)) 


# In[25]:


cnf = confusion_matrix(labels_test, bpred)cnf


# In[26]:


#print(classification_report(labels_test, bpred, target_names=data.lettr))


# In[27]:


plt.figure(figsize=(10,8))
plt.subplot(121)
plt.plot(np.arange(50),acc/len(labels_test)*100)
plt.title('Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.subplot(122)
plt.plot(np.arange(50),err/len(labels_test))
plt.title('Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Error')


# In[28]:


#Implementation of One layer neural networks with SGD + momentum

#Initalize weight and bias using Xaviers Initialization
#Source:https://mnsgrg.com/2017/12/21/xavier-initialization/
hn = 30 #Number of hidden neurons
b1 = -np.sqrt(6/(16+hn)) + 2*np.sqrt(6/(16+hn))*np.random.uniform(size = (hn,1))
w1 = -np.sqrt(6/(16+hn)) + 2*np.sqrt(6/(16+hn))*np.random.uniform(size = (hn,16))
b2 = -np.sqrt(6/(26+hn)) + 2*np.sqrt(6/(26+hn))*np.random.uniform(size = (26,1))
w2 = -np.sqrt(6/(26+hn)) + 2*np.sqrt(6/(26+hn))*np.random.uniform(size = (26,hn))

#Initialize Variables
epoch = 100
lr = 0.1
alp = 0.2
lbl = np.zeros((26,1))
err = np.zeros((epoch,1))
acc = np.zeros((epoch,1))
db1 = np.zeros((hn,1))   #Gradient for bias
dw1 = np.zeros((hn,16))  #Gradient for weight
db2 = np.zeros((26,1))
dw2 = np.zeros((26,hn))

#Temp variables to save best parameters
min_err = np.Inf
max_acc = 0
bpred = np.zeros((len(labels_test),1))

for i in range(epoch):
    shuff = np.random.permutation(len(labels_train))
    #Pass over an epoch
    for j in range(len(labels_train)):
        cls = labels_train.iloc[shuff[j]]
        lbl[ord(cls[0]) - 65] = 1    #One hot encoding

        #Forward Pass
        f = np.reshape(np.array(features_train.iloc[shuff[j]]),(16,1)) #Feature needs to be reshaped due to numpy default behaviour
        l1 = np.matmul(w1,f) + b1
        o1 = 1/(1 + np.exp(-l1)) #Sigmoid
        tot = np.matmul(w2,o1) + b2
        o = softmax(tot)

        #Backward Pass (Might be a little confusing, so try doing the steps on a small matrix)
        db2 = alp*db2 + lr*(o-lbl)
        dw2 = alp*dw2+ lr*np.matmul((o-lbl),np.transpose(o1))
        db1 = alp*db1 + np.transpose(lr*np.matmul(np.transpose(o-lbl),w2))*o1*(1-o1)
        dw1 = alp*dw1 + np.matmul((np.transpose(lr*np.matmul(np.transpose(o-lbl),w2))*o1*(1-o1)),np.transpose(f))

        b1 = b1 - db1
        w1 = w1 - dw1
        b2 = b2 - db2
        w2 = w2 - dw2

        lbl[:] = 0
    
    #Check performance on test set
    for k in range(len(labels_test)):
        cls = labels_test.iloc[k]
        lbl[ord(cls[0]) - 65] = 1
        f = np.reshape(np.array(features_test.iloc[k]),(16,1))

        l1 = np.matmul(w1,f) + b1
        o1 = 1/(1 + np.exp(-l1)) #Sigmoid
        tot = np.matmul(w2,o1) + b2
        o = softmax(tot)

        maxIdx = np.argmax(o)
        pred[k] = chr(ord('A') + maxIdx)

        if pred[k] == cls[0]:
            acc[i] = acc[i] + 1

        entropy = -lbl*np.log(o)
        entropy = np.nan_to_num(entropy)
        err[i] = err[i] + np.sum(entropy)
        lbl[:] = 0

    #print('Epoch:', i, 'Accuracy:', acc[i]/len(labels_test)*100, 'Cross Entropy Error:', err[i]/len(labels_test))
    #Early Stopping condition
    if (err[i] < min_err):
        min_err = err[i]
        bpred = pred

    #Set gradient and labels back to zero
    lbl[:] = 0
    db1[:] = 0
    dw1[:] = 0
    db2[:] = 0
    dw2[:] = 0


# In[36]:


print('One Layer Neural Network:',np.around(np.max(acc)/len(labels_test)*100,decimals=3)) 


# In[30]:


cnf = confusion_matrix(labels_test, bpred)cnf


# In[31]:


#print(classification_report(labels_test, bpred, target_names=data.lettr))


# In[32]:


plt.figure(figsize=(10,8))
plt.subplot(121)
plt.plot(np.arange(100),acc/len(labels_test)*100)
plt.title('Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.subplot(122)
plt.plot(np.arange(100),err/len(labels_test))
plt.title('Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Error')


# In[29]:


#Implementation of 2-layer neural networks with SGD + momentum

#Initalize weight and bias using Xaviers Initialization
#Source:https://mnsgrg.com/2017/12/21/xavier-initialization/
hn1 = 64 #Number of hidden neurons in layer 1
hn2 = 32 #Number of hidden neurons in layer 2
b1 = -np.sqrt(6/(16+hn1)) + 2*np.sqrt(6/(16+hn1))*np.random.uniform(size = (hn1,1))
w1 = -np.sqrt(6/(16+hn1)) + 2*np.sqrt(6/(16+hn1))*np.random.uniform(size = (hn1,16))
b2 = -np.sqrt(6/(hn1+hn2)) + 2*np.sqrt(6/(hn1+hn2))*np.random.uniform(size = (hn2,1))
w2 = -np.sqrt(6/(hn1+hn2)) + 2*np.sqrt(6/(hn1+hn2))*np.random.uniform(size = (hn2,hn1))
b3 = -np.sqrt(6/(26+hn2)) + 2*np.sqrt(6/(26+hn2))*np.random.uniform(size = (26,1))
w3 = -np.sqrt(6/(26+hn2)) + 2*np.sqrt(6/(26+hn2))*np.random.uniform(size = (26,hn2))


#Initialize Variables
epoch = 100
lr = 0.09
alp = 0.008
lbl = np.zeros((26,1))
err = np.zeros((epoch,1))
acc = np.zeros((epoch,1))
db1 = np.zeros((hn1,1))   #Gradient for bias
dw1 = np.zeros((hn1,16))  #Gradient for weight
db2 = np.zeros((hn2,1))
dw2 = np.zeros((hn2,hn1))
db3 = np.zeros((26,1))
dw3 = np.zeros((26,hn2))

#Temp variables to save best parameters
min_err = np.Inf
max_acc = 0
bpred = np.zeros((len(labels_test),1))

for i in range(epoch):
    shuff = np.random.permutation(len(labels_train))
    #Pass over an epoch
    for j in range(len(labels_train)):
        cls = labels_train.iloc[shuff[j]]
        lbl[ord(cls[0]) - 65] = 1    #One hot encoding

        #Forward Pass
        f = np.reshape(np.array(features_train.iloc[shuff[j]]),(16,1)) #Feature needs to be reshaped due to numpy default behaviour
        l1 = np.matmul(w1,f) + b1
        o1 = 1/(1 + np.exp(-l1)) #Sigmoid
        l2 = np.matmul(w2,o1) + b2
        o2 = 1/(1 + np.exp(-l2))
        tot = np.matmul(w3,o2) + b3
        o = softmax(tot)

        #Backward Pass (Might be a little confusing, so try doing the steps on a small matrix)
        #pdb.set_trace()
        db3 = alp*db3 + lr*(o-lbl)
        dw3 = alp*db3 + lr*np.matmul((o-lbl),np.transpose(o2))
        db2 = alp*db2 + np.transpose(lr*np.matmul(np.transpose(o-lbl),w3))*o2*(1-o2)
        dw2 = alp*dw2 + np.transpose(lr*np.matmul(np.transpose(o-lbl),w3))*o2*(1-o2)*np.transpose(o1)
        db1 = alp*db1 + np.transpose(np.matmul(np.transpose(np.transpose(lr*np.matmul(np.transpose(o-lbl),w3))*o2*(1-o2)),w2))*o1*(1-o1)
        dw1 = alp*dw1 + np.matmul(np.transpose(np.matmul(np.transpose(np.transpose(lr*np.matmul(np.transpose(o-lbl),w3))*o2*(1-o2)),w2))*o1*(1-o1),np.transpose(f))

        b1 = b1 - db1
        w1 = w1 - dw1
        b2 = b2 - db2
        w2 = w2 - dw2
        b3 = b3 - db3
        w3 = w3 - dw3

        lbl[:] = 0
    
    #Check performance on test set
    for k in range(len(labels_test)):
        cls = labels_test.iloc[k]
        lbl[ord(cls[0]) - 65] = 1
        f = np.reshape(np.array(features_test.iloc[k]),(16,1))

        l1 = np.matmul(w1,f) + b1
        o1 = 1/(1 + np.exp(-l1)) #Sigmoid
        l2 = np.matmul(w2,o1) + b2
        o2 = 1/(1 + np.exp(-l2))
        tot = np.matmul(w3,o2) + b3
        o = softmax(tot)

        maxIdx = np.argmax(o)
        pred[k] = chr(ord('A') + maxIdx)

        if pred[k] == cls[0]:
            acc[i] = acc[i] + 1

        entropy = -lbl*np.log(o)
        entropy = np.nan_to_num(entropy)
        err[i] = err[i] + np.sum(entropy)
        lbl[:] = 0

    #pdb.set_trace()
    #print('Epoch:', i, 'Accuracy:', acc[i]/len(labels_test)*100, 'Cross Entropy Error:', err[i]/len(labels_test))
    #Early Stopping condition
    if (err[i] < min_err):
        min_err = err[i]
        bpred = pred

    #Set gradient and labels back to zero
    lbl[:] = 0
    db1[:] = 0
    dw1[:] = 0
    db2[:] = 0
    dw2[:] = 0


# In[30]:


print('Two Layer Neural Network:',np.around(np.max(acc)/len(labels_test)*100,decimals=3)) 


# In[31]:


cnf = confusion_matrix(labels_test, bpred)cnf


# In[32]:


#print(classification_report(labels_test, bpred, target_names=data.lettr))


# In[33]:


plt.figure(figsize=(10,8))
plt.subplot(121)
plt.plot(np.arange(100),acc/len(labels_test)*100)
plt.title('Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.subplot(122)
plt.plot(np.arange(100),err/len(labels_test))
plt.title('Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Error')


# The best machine learning models for tabular data like the one shown above are decision trees and random forests. Our neural network with 2 hidden layers does relatively very well on the dataset achieving 95-96% accuracy. Lets see how good this performance is compared to decision trees and random forests.

# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[55]:


clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
ypred = clf.predict(features_test)
acc = np.sum(np.array(ypred) == np.reshape(np.array(labels_test),(-1)))/len(labels_test)
print('Decision Tree: ', np.around(acc*100,3))


# In[56]:


cnf = confusion_matrix(labels_test, ypred)cnf


# In[57]:


#print(classification_report(labels_test, ypred, target_names=data.lettr))


# In[63]:


clf = RandomForestClassifier(n_estimators=50,bootstrap = True)
clf.fit(features_train, labels_train)
ypred = clf.predict(features_test)
acc = np.sum(np.array(ypred) == np.reshape(np.array(labels_test),(-1)))/len(labels_test)
print('Random Forests: ', np.around(acc*100,3))


# In[64]:


cnf = confusion_matrix(labels_test, ypred)cnf


# In[65]:


#print(classification_report(labels_test, bpred, target_names=data.lettr))


# Using just decision trees results in an accuracy of 87% which is poorer than our Nerual Net. However, Random Forests (no of trees - 50) out-perform the neural networks, although by just a small margin. This shows that our networks performance is good enough!  
