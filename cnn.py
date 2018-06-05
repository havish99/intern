	
# coding: utf-8

# In[2]:


from __future__ import division
import os
import keras
from keras.layers import BatchNormalization
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D,GaussianNoise
from keras.models import Sequential
from keras import regularizers
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# In[3]:

np.random.seed(0)
#below code reads images from folders
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg','.JPG','.tif']]):
			img = cv2.imread(os.path.join(folder, filename))
			if img is not None:
				images.append(img)
    return images


# In[4]:


#all calculations related to input
folders = [
    '1',
    '2',
    '3'
]
num_classes=len(folders)
df={}
for folder in folders:
    df[folder] = np.array(load_images_from_folder(folder))
    print(df[folder].shape)
for folder in folders:
    df[folder] = df[folder].reshape(len(df[folder]),128*64*3)
X=[]
y=[]
for i in folders:
	for elem in df[i]:
		X.append(elem)
		y.append(int(i)-1)
X=np.array(X)
y=np.array(y)
X=X/255*1.0
print(np.max(X))
print(X.shape)
print(y.shape)


# # Pre-processing

# In[5]:


num=X.shape[0]
X_reshaped=X.reshape(num,128,64,3)
print(X_reshaped.shape)
#y=keras.utils.to_categorical(y,num_classes)
print(y.shape)


# In[6]:
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.1,random_state=33)

batch_size=8
epochs=20


# # Model

# In[7]:
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test1=keras.utils.to_categorical(y_test,num_classes)
# Model1 = Sequential()
# random.seed(19)


Model1 = Sequential()
random.seed(19)
Model1.add(Conv2D(32, (1, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(128,64,3)))
Model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
Model1.add(BatchNormalization())

Model1.add(Conv2D(32, (3, 1), strides=(1, 1), padding='same', activation='relu'))
Model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
Model1.add(BatchNormalization())

Model1.add(Conv2D(64, (1, 3), padding='same', activation='relu'))
Model1.add(Conv2D(64, (3, 1), padding='same', activation='relu'))
Model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
Model1.add(BatchNormalization())

Model1.add(Conv2D(64, (1, 3), padding='same', activation='relu'))
Model1.add(Conv2D(64, (3, 1), padding='same', activation='relu'))

Model1.add(Conv2D(128, (1, 3), padding='same', activation='relu'))
Model1.add(Conv2D(128, (3, 1), padding='same', activation='relu'))
Model1.add(BatchNormalization())

Model1.add(Conv2D(64,(1,1),padding='same',activation='relu'))

Model1.add(Conv2D(256, (1, 3), padding='same', activation='relu'))
Model1.add(Conv2D(256,(3,1),padding='same',activation='relu'))
Model1.add(BatchNormalization())

Model1.add(Conv2D(128,(1,1),padding='same',activation='relu'))
Model1.add(Conv2D(512, (1, 5), padding='same', activation='relu'))
Model1.add(Conv2D(512,(5,1),padding='same',activation='relu'))
Model1.add(BatchNormalization())

Model1.add(Conv2D(512, (1, 3), padding='same', activation='relu'))
Model1.add(Conv2D(512,(3,1),padding='same',activation='relu'))
Model1.add(BatchNormalization())

Model1.add(GaussianNoise(stddev=0.001))
Model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
Model1.add(BatchNormalization())

Model1.add(Flatten())
Model1.add(Dense(4096, activation='relu'))
Model1.add(Dropout(0.5))
Model1.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
Model1.add(Dropout(0.5))

Model1.add(Dense(num_classes, activation='softmax'))
Model1.summary()
# # Training

# In[18]:



Model1.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=1e-5),metrics=['accuracy'])
Model1.fit(X_train,y_train,batch_size=batch_size,epochs=30,verbose=1)

score=Model1.evaluate(X_test,y_test1)
print(score[0])
print(score[1])
Model1.save("test_2.h5")
predict=Model1.predict_classes(X_test)
print(confusion_matrix(y_test,predict))
target_names=['normal','DR','ARMD']
print(classification_report(y_test,predict,target_names=target_names))
# print(score[0])
# 