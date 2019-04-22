import cv2
from scipy import misc
import matplotlib.pyplot as plt

def extractChannel(channel, name):
    img = misc.imread('img/baboon.png')
    img = img & channel
    img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)
    misc.imsave('img/E2/'+name+'.png', img)

extractChannel(1 << 0, 'baboon-channel0')
extractChannel(1 << 1, 'baboon-channel1')
extractChannel(1 << 2, 'baboon-channel2')
extractChannel(1 << 3, 'baboon-channel3')
extractChannel(1 << 4, 'baboon-channel4')
extractChannel(1 << 5, 'baboon-channel5')
extractChannel(1 << 6, 'baboon-channel6')
extractChannel(1 << 7, 'baboon-channel7')
