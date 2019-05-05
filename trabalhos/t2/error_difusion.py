import numpy as np
import cv2

def zigZagRange(flag, i, comp):
    if(flag == False):
        return [1, comp - 2, 1]
    else:
        if(i % 2 != 0):
            return [1, comp - 2, 1]
        else:
            return [comp - 2, 1, -1]


def pontilhadoPorDifusaoDeErro(caminho, zigzag, name):
    aux = cv2.imread(caminho, -1) # 0 determina que a img é lida em tons de cinza
    img = cv2.copyMakeBorder(aux, 1, 1, 1, 1, cv2.BORDER_CONSTANT,value=0) #aplicação de padding zero com largura 1

    

    a = 1
    incr = 1
    b = len(img) - 2 # compensação do padding

    alt = np.size(img, 0)
    comp = np.size(img, 1)

    for i in range(1, alt-1):
        
        a, b, incr = zigZagRange(zigzag, i, comp)

        for j in range(a, b, incr):
            pixel = img[i][j]

            if(pixel < 128):
                pixel = 0
            else:
                pixel = 255
            
            erro = img[i][j] - pixel

            img[i][j+1] = img[i][j+1] + (7/16)*erro
            img[i+1][j-1] = img[i+1][j-1] + (3/16)*erro
            img[i+1][j] = img[i+1][j+1] + (5/16)*erro
            img[i+1][j+1] = img[i+1][j+1] + (1/16)*erro

            img[i][j] = pixel
    
    cv2.imwrite('imgRes/'+name+'.pbm', img)



pontilhadoPorDifusaoDeErro('../img/monarch.pgm', True, 'monarchzigzag')
pontilhadoPorDifusaoDeErro('../img/monarch.pgm', False, 'monarchtnotzigzag')