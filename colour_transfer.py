import numpy as np
import argparse
import cv2

def color_transfer(source, target):
	"""
	Transfers the color distribution from the source to the target
	image using the mean and standard deviations of the L*a*b*
	color space.
	This implementation is (loosely) based on to the "Color Transfer
	between Images" paper by Reinhard et al., 2001.
	Parameters:
	-------
	source: NumPy array
		OpenCV image in BGR color space (the source image)
	target: NumPy array
		OpenCV image in BGR color space (the target image)
	Returns:
	-------
	transfer: NumPy array
		OpenCV image (w, h, 3) NumPy array (uint8)
	"""
	# convert the images from the RGB to L*ab* color space, being
	# sure to utilizing the floating point data type (note: OpenCV
	# expects floats to be 32-bit, so use that instead of 64-bit)
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

	# compute color statistics for the source and target images
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

	# subtract the means from the target image
	(l, a, b) = cv2.split(target)
	l -= lMeanTar
	a -= aMeanTar
	b -= bMeanTar

	# scale by the standard deviations
	l = (lStdTar / lStdSrc) * l
	a = (aStdTar / aStdSrc) * a
	b = (bStdTar / bStdSrc) * b

	# add in the source mean
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc

	# clip the pixel intensities to [0, 255] if they fall outside
	# this range
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)

	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
	
	# return the color transferred image
	return transfer

def show_image(title, image, width = 300):
	# resize the image to have a constant width, just to
	# make displaying the images take up less screen real
	# estate
	r = width / float(image.shape[1])
	dim = (width, int(image.shape[0] * r))
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

	# show the resized image
	cv2.imshow(title, resized)


# load the images
source = cv2.imread('source.jpg')
target = cv2.imread('target.jpg')

# transfer the color distribution from the source image
# to the target image
transfer = color_transfer(source, target)

cv2.imwrite('transfer.jpg', transfer)