import numpy as np
import cv2
#from matplotlib import pyplot as plt

img = cv2.imread('../img/city.png', 0)
f = np.fft.fft2(img) # Transformada de fourier rápida
fshift = np.fft.fftshift(f) #Move o espectro para o centro
magnitude_spectrum = 20*np.log(np.abs(fshift)) # operação para vizualização

#Geração do filtro gaussiano
gauss = cv2.getGaussianKernel(img.shape[0], 30)
gauss = gauss * gauss.T

#Aplica o filtro gaussiano a imagem no domínio da frequência
imgWithgaussianFilt = fshift * gauss

#Visualiação 
#magnitude_spectrum = 20*np.log(np.abs(fshift))
magnitude_spectrum2 = 20*np.log(np.abs(imgWithgaussianFilt))

#plt.subplot(121),plt.imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(121),plt.imshow(magnitude_spectrum, cmap = 'gray')
#plt.title('Input FFT Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(magnitude_spectrum2, cmap = 'gray')
#plt.title('Gaussian Filter'), plt.xticks([]), plt.yticks([])
#plt.show()

#Devolve a imagem para o domínio espacial
res = np.fft.ifftshift(imgWithgaussianFilt)
res = np.fft.ifft2(res)
res = np.abs(res)

#Normaliza entre 0 - 255 
res = cv2.normalize(res, None, 0, 255, norm_type=cv2.NORM_MINMAX)

cv2.imwrite("imgRes/E2/img_with_gauss.png", res) # resultado 
cv2.imwrite("imgRes/E2/FFT.png",magnitude_spectrum) # espectro original
cv2.imwrite("imgRes/E2/FFT_WITH_GAUSS.png",magnitude_spectrum2) # espectro com aplicação de gauss