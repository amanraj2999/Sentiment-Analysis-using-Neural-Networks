#!/usr/bin/env python
# coding: utf-8

# In[51]:


from keras.datasets import imdb


# ### Dataset preparation

# In[52]:


((XT,YT),(Xt,Yt)) = imdb.load_data(num_words=10000)


# In[53]:


len(XT)                    #X train dataset


# In[54]:


len(Xt)                    #X Test dataset


# In[55]:


print(XT[0])


# In[56]:


word_idx = imdb.get_word_index()


# In[57]:


print(word_idx.items())


# In[58]:


idx_word = dict([value,key] for (key,value) in word_idx.items())


# In[59]:


print(idx_word.items())


# In[60]:


actual_review = ' '.join([idx_word.get(idx-3,'?') for idx in XT[0]])


# In[61]:


print(actual_review)


# In[62]:


## Next Step - Vectorize the Data
## Vocab Size - 10,000 We will make sure every sentence is represented by a vector of len 10000 [00000111.....00101010]
import numpy as np
def vectorize_sentences(sentences,dim=10000):
  
  outputs = np.zeros((len(sentences),dim))
  
  for i,idx in enumerate(sentences):
    outputs[i,idx] = 1
   
  return outputs


# In[63]:


X_train = vectorize_sentences(XT)
X_test = vectorize_sentences(Xt)
print(X_train.shape)
print(X_test.shape)


# In[64]:


print(X_train[0])


# In[65]:


Y_train = np.asarray(YT).astype('float32')
Y_test = np.asarray(Yt).astype('float32')


# ### Defining model architecture
# * Use Fully Connected/Dense Layers with RelU Activation
# * 2 Hidden Layers with 16 units each
# * 1 Output Layer with 1 unit (Sigmoid Activation)

# In[66]:


from keras import models
from keras.layers import Dense


# In[67]:


# Define the model
model = models.Sequential()
model.add(Dense(16,activation='relu',input_shape=(10000,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[68]:


# Compile the Model
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])


# In[69]:


model.summary()


# ### Training & Validation

# In[70]:


x_val = X_train[:5000]
x_train_new = X_train[5000:]

y_val = Y_train[:5000]
y_train_new = Y_train[5000:]


# In[71]:


hist = model.fit(x_train_new,y_train_new,epochs=20,batch_size=512,validation_data=(x_val,y_val))


# ### Visualising results 

# In[89]:


import matplotlib.pyplot as plt


# In[90]:


h = hist.history


# In[94]:


# Visualising Validation loss vs Training Loss

plt.plot(h['val_loss'],label="Validation Loss")
plt.plot(h['loss'],label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[95]:


# Visualising Validation Acc vs Training Acc

plt.plot(h['val_accuracy'],label="Validation acc")  
plt.plot(h['accuracy'],label="Training Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[76]:


model.evaluate(X_test,Y_test)[1]


# In[77]:


model.evaluate(X_train,Y_train)[1]


# In[78]:


model.predict(X_test)      # gives binary prediction --> closer to 1 --> positive & closer to 0 --> negative 

