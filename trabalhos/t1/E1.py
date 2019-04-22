import numpy as np
import cv2

h1 =  np.array([[0, 0, -1, 0, 0],
           [0, -1, -2, -1, 0],
           [-1, -2, 16, -2, -1],
           [0, -1, -2, -1, 0],
           [0, 0, -1, 0, 0]])

h2 = np.array([[1, 4, 6, 4, 1],
               [4, 16, 24, 16, 4],
               [6, 24, 36, 24, 6],
               [4, 16, 24, 16, 4],
               [1, 4, 6, 4, 1]])/256

h3 = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

h4 = h3.T # O filtro h4 pode ser gerado pela transposta de h3

# Aplica filtro: recebe o caminho da img, um filtro e o nome para salvar a imagem
def filterApplier(imgPath, filter, name):
    img =  cv2.imread(imgPath, 0) # 0 determina que a imagem está em escala de cinza
    res = cv2.filter2D(img, -1, filter)
    res = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite('imgRes/E1/'+name+'.png', res)
#--------------------------------------------------------------------------------

# Combina filtros: faz a combinação dos resultado da aplicação de dois filtros na mesma imagem
def combineFilters(imgPath, filter1, filter2, name):
    img =  cv2.imread(imgPath, 0) # 0 determina que a imagem está em escala de cinza
    res1 = cv2.filter2D(img, -1, filter1)
    res2 = cv2.filter2D(img, -1, filter2)
    res = (res1**2 + res2**2)**0.5 # forma da combinação do resultado dos filtros
    res = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite('imgRes/E1/'+name+'.png', res)

filterApplier('city.png', h1, 'h1') # Aplica filtro do item a (h1 - passa alta)
filterApplier('city.png', h2, 'h2') # Aplica filtro do item b (h2 - passa baixa)
filterApplier('city.png', h3, 'h3') # Aplica filtro do item c (h3 - componente vertical de sobel)
filterApplier('city.png', h4, 'h4') # Aplica filtro do item c (h4 - componente horizontal de sobel) 
combineFilters('city.png', h3, h4, 'h5') # Aplica filtros do item c e combina os resultados (h3 e h4)
