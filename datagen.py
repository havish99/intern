from keras.preprocessing.image import ImageDataGenerator as IDG
import numpy as np
import os
from scipy.misc import imread

datagen=IDG(horizontal_flip=True,vertical_flip=True)
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir('1'):
#         if any([filename.endswith(x) for x in ['.jpeg', '.jpg','.JPG']]):
#             img = Image.open(os.path.join(folder, filename))
#             img=np.array(img)
#             datagen.fit(img.reshape(1,1696,2544,3))
#             i=0
# 		for batch in datagen.flow(img.reshape(1,1696,2544,3), batch_size=1,save_to_dir='preview',
#                            save_prefix='lol', save_format='jpeg'):
#   			i += 1
#     		if i == 4:
#         break
#             if img is not None:
#                 images.append(img)
#     return images



for root, dirs, files in os.walk('3'):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".JPG"):
            img = imread(os.path.join(root, file))
            k=img.shape
            i=0
            for batch in datagen.flow(img.reshape(1,k[0],k[1],k[2]), batch_size=1,save_to_dir='preview',save_prefix='lol', save_format='jpeg'):
            	i=i+1
            	if i==2:
            		break
   