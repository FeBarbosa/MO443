import cv2
import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((8, 8)) # original image
img2 = np.zeros((8, 8)) # image with median filter applied

# Generate the image
for i in range(8):
    for j in range(8):
        img[i][j] = abs(i-j)
        img2[i][j] = abs(i-j)

# Median filter application
for i in range(1,7):
    v = np.zeros((3,3)) #temporary array to median
    for j in range(1,7):
        v[0][0] = img[i-1][j-1]
        v[1][0] = img[i][j-1]
        v[2][0] = img[i+1][j-1]
        v[0][1] = img[i-1][j]
        v[1][1] = img[i][j]
        v[2][1] = img[i+1][j]
        v[0][2] = img[i-1][j+1]
        v[1][2] = img[i][j+1]
        v[2][2] = img[i+1][j+1]
        img2[i][j] = np.median(v) #Median

fig = plt.figure()

a = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(img)
a.set_title('Original')

a = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img2)
a.set_title('median')

plt.show()