from keras.models import load_model
import cv2
import numpy as np
import os
import keras
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg','.JPG']]):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

folders=['1','2']
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

num_classes=2
num=X.shape[0]
X_reshaped=X.reshape(num,128,64,3)
print(X_reshaped.shape)
y=keras.utils.to_categorical(y,num_classes)
print(y.shape)


# In[6]:





model = load_model('mymodel2.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

classes=model.predict_classes(X_reshaped)

