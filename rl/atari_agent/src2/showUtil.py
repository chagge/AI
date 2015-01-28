import cv2
import numpy as np
import sys

img = np.zeros((84, 84), np.float32)
u = v = 0
i = 0
while(i < 84*84-1):
	img[u,v] = (input())
	if(v == 83):
		u = u + 1
		garb=raw_input()
	v = (v+1)%84
	i = i + 1

#vis2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.imwrite( 'frame.png', img)