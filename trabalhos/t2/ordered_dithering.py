import numpy as np
import cv2

mask1 = np.array([[6, 8, 4],
                  [1, 0, 3],
                  [5, 2, 7]])

mask2 = np.array([[0, 12, 3, 15],
                  [8, 4, 11, 7],
                  [2, 14, 1, 13],
                  [10, 6, 9, 5]])

def pontilhadoOrdenado(caminho, mascara, name):
    img = cv2.imread(caminho, 0) # 0 para indicar img em tons de cinza

    sizeMask = np.size(mascara, 0) # dimensao da mascara
    maxValueNorm = sizeMask * sizeMask

    img = cv2.normalize(img, None, 0, maxValueNorm, norm_type = cv2.NORM_MINMAX) #normalizacao

    alt = np.size(img, 0)  # qt de linhas da img
    comp = np.size(img, 1) # qt de colunas da img

    img2 = np.full((alt*sizeMask, comp*sizeMask), 255,dtype = np.uint8) # resultado


    for i in range(0, alt):
        for j in range(0, comp):
            pixel = img[i, j]

            for k in range(0, sizeMask):
                for l in range(0, sizeMask):
                    if(pixel < mascara[k][l]):
                        img2[i*sizeMask + k][j*sizeMask + l] = 0

    
    cv2.imwrite('imgRes/ordered_dithering/'+name+'.pbm', img2)



pontilhadoOrdenado('../img/rock.pgm', mask1, 'rock3x3')
pontilhadoOrdenado('../img/rock.pgm', mask2, 'rock4x4')

