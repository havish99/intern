from __future__ import division
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import cv2
import keras


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg','.JPG']]):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

folders = [
    '1',
    '2'
]
num_classes=2
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
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2,random_state=33)

batch_size=16
epochs=20
num_classes=2

# # Model

# In[7]:
y_train=keras.utils.to_categorical(y_train,num_classes)




model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (128, 64, 3))

# for layer in model.layers[:5]:
#     layer.trainable = False


x = model.output
x = Flatten()(x)
x = Dense(2048, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(2048, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)
model_final.summary()
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=1e-5), metrics=["accuracy"])

model_final.fit(X_train,y_train,batch_size=batch_size,epochs=22,verbose=1)
model_final.save("transfer.h5")
predict=model_final.predict(X_test)
predict=np.argmax(predict,axis=1)
print(confusion_matrix(y_test,predict))