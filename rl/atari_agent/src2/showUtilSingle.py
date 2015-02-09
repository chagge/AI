import cv2
import numpy as np
import sys

for kk in range(0,1):
	img = np.zeros((84, 84), np.float32)
	for u in range(84):
		for v in range(84):
			img[u,v] = input()

	#vis2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.imwrite( 'frame_' + str(kk) +'_.png', img)