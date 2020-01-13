#!/usr/bin/env python
# coding: utf-8

# In[220]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# In[223]:


dataDir = "C:/Users/DHANUJA/Downloads/basicshapes/shapes"
categories = ["circles","triangles","squares"]
train_data = []
test_data = []
IMG_SIZE=30

for category in categories:
    count=0
    path = os.path.join(dataDir,category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            if count<180:
                train_data.append([img_array, categories.index(category)])
                count=count+1
            else:    
                test_data.append([img_array, categories.index(category)])
        except Exception as e:
            pass


# In[104]:


print(len(train_data))
print(len(test_data))


# In[214]:


import random

random.shuffle(train_data)
random.shuffle(test_data)
random.shuffle(train_data)
random.shuffle(test_data)
random.shuffle(train_data)
random.shuffle(test_data)

x_train=[]
y_train=[]
x_test=[]
y_test=[]
for feature,label in train_data:
    x_train.append(feature)
    y_train.append(label)
for feature,label in test_data:
    x_test.append(feature)
    y_test.append(label)
    
    

import tensorflow as tf
model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(784),
      tf.keras.layers.Activation('sigmoid'),
      #tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(3),
      tf.keras.layers.Activation('sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x_train=np.array(x_train).reshape(-1, 28, 28)
x_test=np.array(x_test).reshape(-1, 28, 28)

y_train=np.array(y_train)
y_test=np.array(y_test)
#from tensorflow.keras.utils import to_categorical

x_train, x_test = x_train / 255.0, x_test / 255.0
#y_train, y_test = y_train / 2.0, y_test / 2.0
model.fit(x_train, y_train, epochs=20)


# In[215]:


model.evaluate(x_train,  y_train)


# In[216]:


num=9
print(y_test[num])
plt.imshow(x_test[num])
plt.show()
predictions = model.predict([x_test])
print(np.argmax(predictions[num]))
print(predictions)


# In[217]:


model.summary()


# In[218]:


model.evaluate(x_test,  y_test)


# In[ ]:




