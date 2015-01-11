import cv2
import numpy as np
import sys

fname = sys.argv[1]
with open(fname) as f:
	content = f.readlines()
line = content[0]
img = np.zeros((210, 160), np.float32)
u = v = 0
i = 0
while(i < 2*160*210):
	x = str(line[i]) + str(line[i+1])
	y = int(x,16)
	img[u,v] = y
	if(v == 159):
		u = u + 1
	v = (v+1)%160
	i = i + 2

#vis2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.imwrite( 'hi.png', img)