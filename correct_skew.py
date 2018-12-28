# USAGE
# python correct_skew.py --image images/neg_28.png

# import the necessary packages
# import np as np
import numpy as np
import argparse
import cv2
import os
from datetime import datetime

def getDeskewedFilename(srcName, id=None):
	filename_w_ext = os.path.basename(srcName)
	filename, file_extension = os.path.splitext(filename_w_ext)
	hms = datetime.now().strftime('%d%H%M%S')
	# newName = ''
	if id != None:
		newName = srcName.replace(filename, 'de'+filename+"_"+id+"_"+hms)
	else:
		newName = srcName.replace(filename, 'de'+filename+"_"+hms)
	return newName

# construct the argument parse and parse the argumentss
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image file")
print("ap: {}".format(ap))
args = vars(ap.parse_args())
print("args: {}".format(args))

# load the image from disk
filename = args["image"]
print('filename:{}'.format(filename))
image = cv2.imread(filename)

# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

# threshold the image, setting all foreground pixels to 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
print(type(thresh))
cv2.imwrite(getDeskewedFilename(filename, "thresh"), thresh)

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
print("angle: {}".format(angle))

# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = -(90 + angle)

# otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = -angle

# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# save rotated image to new file
cv2.imwrite(getDeskewedFilename(filename), rotated)
height, width, channels = rotated.shape
print("Heiht:{}, Width:{}, Channels:{}".format(height, width, channels))
dim2 = rotated.reshape(channels, height,width)
print(dim2)

# draw the correction angle on the image so we can validate it
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
print("[INFO] angle: {:.5f}".format(angle))
cv2.imshow("Input", image)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
