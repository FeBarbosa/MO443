import cv2 
import numpy as np
import matplotlib.pyplot as plt
import copy

kernel1mod = np.ones((1, 40), np.uint8)
kernel2mod = np.ones((60, 1), np.uint8)
kernel3mod = np.ones((1, 10), np.uint8)
kernel4mod = np.ones((8, 1), np.uint8)

kernel1original = np.ones((1, 100), np.uint8)
kernel2original = np.ones((200, 1), np.uint8)
kernel3original = np.ones((1, 30), np.uint8)

# quando line = True a função só calcula a quantidade de linhas e não produz nenhuma imagem como resultado
def textRecognation(path, name, kernel1, kernel2, kernel3, kernel4 = None, line = False):

    # Leitura e binarização (normalização)
    img = cv2.imread(path, 0)
    img = img/255
    img = 1-img

    # Passos (1)-(2) -> (1): dilatação seguida de erosão -> Fechamento
    closing1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)

    # Passos (3)-(4) -> (2): dilatação seguida de erosão -> Fechamento
    closing2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)

    # Passo (5) -> (3): Interseção entre (2) e (4)
    intersection = cv2.bitwise_and(closing1, closing2)

    # Passo (6) -> (4): Fechamento com elemento 1x30
    closing3 = cv2.morphologyEx(intersection, cv2.MORPH_CLOSE, kernel3)

    # Passo Extra: utilizado para a separação dos componentes conexos diretamente em palavras
    closing3 = cv2.dilate(closing3, kernel4)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel5)

    # Passo (7) - (5): Identificação de componentes conexos
    closing3 = closing3.astype(np.uint8)
    ret, labels = cv2.connectedComponents(closing3)

    for label in range(1, ret):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255

    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in cnts]

    comps = img
    compsAll = copy.copy(img)
    c = 0

    # Passo (8) -> (6): Para cada componente Conexo
        # (a) Razão entre os pixels preto e o número total de pixels
        # (b) Razão entre o número de transições verticais e horizontias branco para preto e o número total de pixels
    for i in range(len(boxes)):
        y, x, h, w = boxes[i] # coordenadas de uma caixa
        p = 0
        vertical_change = 0
        horizontal_change = 0
        tmp = img[x:x+w, y:y+h]

        for j in range(x, x+w):
            for k in range(y, y+h):
                if(1-img[j][k]) == 0:
                    p += 1
                if(j < img.shape[0]):
                    vertical_change += abs(img[j][k]-img[j+1][k])
                if(k < img.shape[1]):
                    horizontal_change += abs(img[j][k]-img[j][k+1])
            
        # Passo (9) -> (7): Classificação dos componentes conexos em texto e não texto
        #Passo (10) -> (8): Seguimentação de linhas e palavras e cálculo das respectivas quantidades

        cv2.rectangle(compsAll, (y, x),(y+ h, x+w), 1)

        if((p/(h*w) > 0.10) and (p/(h*w) < 0.70)):
            cv2.rectangle(comps, (y, x),(y+ h, x+w), 1)
            c = c+1


    if(line == False):
        print('Número de Letras: '+str(c))
        cv2.imwrite('imgRes/06-'+name+'-closing.pbm', (1-closing3)*255)
        cv2.imwrite('imgRes/07-'+name+'-comps.pbm', (1-comps)*255)
        cv2.imwrite('imgRes/07-'+name+'-compsAll.pbm', (1-compsAll)*255)
    else:
        print('Número de Linhas: '+str(c))
   
textRecognation('../img/bitmap.pbm', 'bitmap-words', kernel1mod, kernel2mod, kernel3mod, kernel4mod)
textRecognation('../img/bitmap.pbm', 'bitmap-line', kernel1original, kernel2original, kernel3original, kernel4mod, True)