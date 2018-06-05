from PIL import Image
import glob

i=0

for img in glob.glob("*.jpg"):
	im=Image.open(img)
	im=im.resize((100,100),Image.NEAREST)
	im.save(img)
	i=i+1

