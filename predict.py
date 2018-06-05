from keras.models import load_model
import cv2
import numpy as np
import os
model = load_model('mymodel2.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg','.JPG']]):
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            if img is not None:
                images.append(img)
    return images




img = load_images_from_folder('preview') 
img=np.array(img)
img=img/255.0
k=img.shape
img=img.reshape(k[0],k[2],k[1],1)
# img=cv2.imread("lol_0_780.jpeg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# img=cv2.resize(img,(128,64))
# print(img.shape)
# img=img.reshape(1,128,64,1)
# img=img/255.0	
classes = model.predict_classes(img)
k=0
for i in classes:
	if i==0:
		k=k+1
print(k)