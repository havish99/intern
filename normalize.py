from PIL import Image
import numpy as np
import cv2
img=np.array(Image.open("Annapurna_Kanchgarla_16-04-1961_P750197_(0000).jpg"))

tot=img[:,:,0]+img[:,:,1]+img[:,:,2]+0.00001

img[:,:,0]=img[:,:,0]/tot
img[:,:,1]=img[:,:,1]/tot
img[:,:,2]=img[:,:,2]/tot

img=img*255
cv2.imwrite('lol.jpg',img)

