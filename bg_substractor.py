import numpy as np
import cv2
from matplotlib import pyplot as plt


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


# img = cv2.imread('./pic/man_small.jpg')
# img = cv2.imread('./pic/man_small_inverted.jpg')
img = cv2.imread('./pic/man_extra_small.jpg')
# img = cv2.imread('./pic/man_extra_small_inverted.jpg')


img = adjust_gamma(img, 3.0)
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# 1200 x 1600 - small
# rect = (400,600,400,400)
# 720 x 960 - extra small
# cv2.rectangle(img,(260,330),(480,630),(0,255,0),3)
rect = (260,330,220,300)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.show()
