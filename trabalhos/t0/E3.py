import numpy as np
import cv2
from scipy import misc
import matplotlib as plt

# img = misc.imread('img/baboon.png')

img = np.zeros((8,8), dtype=np.int64)

x = np.zeros((16,2), dtype=np.int16)

k = 0

step = int(len(img)/4)
print(step)

for i in range(0, len(img), step):
    for j in range(0, len(img), step):
        x[k][0] = i
        x[k][1] = j
        k = k+1

print(x)