import cv2 
import numpy as np
import matplotlib.pyplot as plt

kernel1mod = np.ones((1, 40), np.uint8)
kernel2mod = np.ones((60, 1), np.uint8)
kernel3mod = np.ones((1, 10), np.uint8)
kernel4mod = np.ones((8, 1), np.uint8)

kernel1original = np.ones((1, 100), np.uint8)
kernel2original = np.ones((200, 1), np.uint8)
kernel3original = np.ones((1, 30), np.uint8)

def textRecognation(path, name, kernel1, kernel2, kernel3, kernel4 = None):

    # Leitura e binarização (normalização)
    img = cv2.imread(path, 0)
    img = img/255
    img = 1-img

    # Passos (1)-(2): dilatação seguida de erosão com elemento 1x100
    dilatation1 = cv2.dilate(img, kernel1)
    erosion1 = cv2.erode(dilatation1, kernel1)

    # Passos (3)-(4): dilatação seguida de erosão com elemento 1x200
    dilatation2 = cv2.dilate(img, kernel2)
    erosion2 = cv2.erode(dilatation2, kernel2)

    # Passo (5): Interseção entre (2) e (4)
    intersection = cv2.bitwise_and(erosion1, erosion2)

    # Passo (6): Fechamento com elemento 1x30
    closing = cv2.morphologyEx(intersection, cv2.MORPH_CLOSE, kernel3)

    # Passo Extra: utilizado para a separação dos componentes conexos diretamente em palavras
    closing = cv2.dilate(closing, kernel4)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel5)

    # Passo (7): Identificação de componentes conexos
    closing = closing.astype(np.uint8)
    ret, labels = cv2.connectedComponents(closing)

    # Mostrar componentes com cores distintas
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Conversão de sistema de cores para exibição
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    # cv2.imwrite('imgRes/07-'+name+'-labeled_components.png', labeled_img)

    for label in range(1, ret):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255

    cv2.imwrite('imgRes/07-'+name+'-test.png', mask)

    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in cnts]

    # print(cnts[0])
    # print(hierarchy[0])

    # test = (1-img)*255
    # cv2.drawContours(test, cnts, -1, (0, 255, 0))
    # cv2.imwrite('imgRes/00-'+name+'-test.png', test)

    # Converte img para tres canais para exibição das caixas na imagem
    img3 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img3[:, :, 0] = img
    img3[:, :, 1] = img
    img3[:, :, 2] = img

    area = np.array([None for i in range(len(boxes))])

    comps = (1-img3)*255
    c = 0

    # Passo (8): Para cada componente Conexo
        # (a) Razão entre os pixels preto e o número total de pixels
    for i in range(len(boxes)):
        y, x, h, w = boxes[i]
        p = 0
        b = 0
        change_v = 0
        change_h = 0
        tmp = img[x:x+w, y:y+h]

        for j in range(x, x+w):
            for k in range(y, y+h):
                if(1-img[j][k]) == 0:
                    p += 1
                if(j < img.shape[0]):
                    change_v += abs(img[j][k]-img[j+1][k])
                if(k < img.shape[1]):
                    change_h += abs(img[j][k]-img[j][k+1])
            
        area[i] = (p/(h*w), (change_v+change_h))
        
        # Regra
        if(p/(h*w) > 0.15):
            cv2.rectangle(comps, (y, x),(y+ h, x+w), (0, 0, 255))
            c = c+1

        # (b) Razão entre transições horizontais e verticais de pixels pretos e brancos e o número total de pixels pretos   

    area = np.zeros((len(boxes)))
    for i in range(len(boxes)):
        y,x,h,w = boxes[i]
        p = 0
        b = 0
        for j in range(x,x+w):
            for k in range(y,y+h):
                if(1-img[j][k]) == 0:
                    p += 1
        area[i] = p/(h*w)

    # Passo (9): Classificação dos componentes conexos em texto e não texto

    #Passo (10): Seguimentação de linhas e palavras e cálculo das respectivas quantidades

    # cv2.imwrite('imgRes/01-'+name+'-dilated1.pbm', 1-dilatation1)
    # cv2.imwrite('imgRes/02-'+name+'-eroded1.pbm', 1-erosion1)
    # cv2.imwrite('imgRes/03-'+name+'-dilated2.pbm', 1-dilatation2)
    # cv2.imwrite('imgRes/04-'+name+'-eroded2.pbm', 1-erosion2)
    # cv2.imwrite('imgRes/05-'+name+'-intersection.pbm', 1-intersection)
    # cv2.imwrite('imgRes/06-'+name+'-closing.pbm', 1-closing)
    # cv2.imwrite('imgRes/07-'+name+'-comps.png', comps)
   
textRecognation('../img/bitmap.pbm', 'bitmapline', kernel1mod, kernel2mod, kernel3mod, kernel4mod)
textRecognation('../img/bitmap.pbm', 'bitmap', kernel1original, kernel2original, kernel3original)