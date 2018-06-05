
import numpy as np
import os
import cv2
from sklearn import preprocessing,cross_validation,svm
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
    '2',
]

df={}
for folder in folders:
    df[folder] = np.array(load_images_from_folder(folder))
    df[folder] = df[folder].reshape(len(df[folder]),224*224*3)
X=[]
y=[]
for i in folders:
	for elem in df[i]:
		X.append(elem)
		y.append(int(i)-1)
X=np.array(X)
y=np.array(y)
print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf=svm.SVC()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print(accuracy)