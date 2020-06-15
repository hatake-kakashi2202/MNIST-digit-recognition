#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# In[6]:


data,target = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)


# In[7]:


X = data.astype(np.float32)
X /= 255.
print(X.shape)
encoder = OneHotEncoder(sparse=False,categories='auto')
y = encoder.fit_transform(target.reshape(-1, 1))
print(y.shape)


# In[8]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=4)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[187]:


n_x = 784
n_h1 = 512
n_h2 = 256
n_h3 = 128
n_y = 10
l_rate = 1e-4
iters = 1000
b_size = 16
dropout = 0.7


# In[188]:


X = tf.placeholder("float", [None, n_x])
Y = tf.placeholder("float", [None, n_y])
keep_prob = tf.placeholder(tf.float32)


# In[189]:


def initialize_parameters(n_x, n_h1,n_h2,n_h3, n_y):
    #     n_x = 784
    #     n_h1 = 512
    #     n_h2 = 256
    #     n_h3 = 128
    #     n_y = 10
    np.random.seed(1)
    
    W1 = tf.Variable(tf.truncated_normal([n_x, n_h1], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[n_h1]))
    W2 = tf.Variable(tf.truncated_normal([n_h1, n_h2], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[n_h2]))
    W3 = tf.Variable(tf.truncated_normal([n_h2, n_h3], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[n_h3]))
    W4 = tf.Variable(tf.truncated_normal([n_h3, n_y], stddev=0.1))
    b4 = tf.Variable(tf.constant(0.1, shape=[n_y]))

    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,}
    
    return parameters


# In[190]:


parameters = initialize_parameters(n_x, n_h1,n_h2,n_h3, n_y)
def linear_forward(X,parameters):
    Z=[]
    Z.append(tf.add(tf.matmul(X, parameters['W1']), parameters['b1']))
    Z.append(tf.add(tf.matmul(Z[-1], parameters['W2']), parameters['b2']))
    Z.append(tf.add(tf.matmul(Z[-1], parameters['W3']), parameters['b3']))
    layer_drop = tf.nn.dropout(Z[-1], rate=1-keep_prob)
    Z.append(tf.matmul(Z[-1], parameters['W4']) + parameters['b4'])
    return Z


# In[191]:


Z = linear_forward(X,parameters)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=Z[-1]
        ))
train_step = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)


# In[192]:


correct_pred = tf.equal(tf.argmax(Z[-1], 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[193]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[196]:


x=0
y=b_size
for i in range(iters):
    
    batch_x, batch_y = X_train[x:y],y_train[x:y]
    x+=b_size
    y+=b_size
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
        })


# In[197]:


test_accuracy = sess.run(accuracy, feed_dict={X: X_test, Y: y_test, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)


# In[ ]:





# In[ ]:




