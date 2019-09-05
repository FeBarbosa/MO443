import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

def imgPanoramic(path1, path2, name, choice = 0):
        # PASSO (1): Ler e converter imagens em escala de cinza
        imgOrigin1 = cv2.imread(path1)
        img1 = cv2.cvtColor(imgOrigin1, cv2.COLOR_BGR2GRAY) # RGB to Grayscale

        imgOrigin2 = cv2.imread(path2)
        img2 = cv2.cvtColor(imgOrigin2, cv2.COLOR_BGR2GRAY) # RGB to Grayscale

        # PASSO (2): Encontrar pontos de interesse e descritores
        if(choice == 0):
                func = cv2.xfeatures2d.SIFT_create()
        elif(choice == 1):
                func = cv2.xfeatures2d.SURF_create()
        elif(choice == 2):
                func = cv2.ORB_create()

        # keypoints and descriptors for both imagens
        keyP1, descr1 = func.detectAndCompute(img1, None)
        keyP2, descr2 = func.detectAndCompute(img2, None)

        # PASSO (3): calcular distâncias
        # using Brute-Force Matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descr1, descr2, k = 2)

        # PASSO (4): selecionar as melhores correspondências
        good = []
        for m, n in matches:
                if m.distance < 0.75 * n.distance:
                        good.append([m])
        
        match = np.asarray(good)

        if len(match[:,0]) >= 4:
                source = np.float32([ keyP1[m.queryIdx].pt for m in match[:,0] ]).reshape(-1,1,2)
                target = np.float32([ keyP2[m.trainIdx].pt for m in match[:,0] ]).reshape(-1,1,2)

        # PASSO (5): estimar a matriz de homografia com RANSAC
        homo, masked = cv2.findHomography(source, target, cv2.RANSAC, 5.0)

        print(np.shape(imgOrigin1))
        print(np.shape(imgOrigin2))

        # PASSO (6): Alinhar as imagem com uma projeção de perspectiva
        img3 = cv2.warpPerspective(imgOrigin1, homo, (imgOrigin2.shape[1] + imgOrigin2.shape[1], imgOrigin2.shape[0]))

        height = max(imgOrigin1.shape[0], imgOrigin2.shape[0])
        width = max(imgOrigin1.shape[1], imgOrigin2.shape[1])

        # PASSO (7): Unir as imagens criando uma imagem panorâmica      
        # img3[0:imgOrigin1.shape[0], 0:imgOrigin1.shape[1]] = imgOrigin2
        img3[0:height, 0:width] = imgOrigin2
        cv2.imwrite('target'+name+'.jpg', img3)

        # PASSO (8): Desenhar retas entre os pontos correspondentes entre as imagens

        img4 = np.zeros(np.shape(img1), np.uint8)
        img4 = cv2.drawMatchesKnn(imgOrigin2, keyP2, imgOrigin1, keyP1, good[:100], img3, flags=2)
        cv2.imwrite('matches-'+name+'.jpg', img4)

imgPanoramic('../img/foto1A.jpg', '../img/foto1B.jpg', 'foto1-sift', 0)
imgPanoramic('../img/foto1A.jpg', '../img/foto1B.jpg', 'foto1-surf', 1)
imgPanoramic('../img/foto1A.jpg', '../img/foto1B.jpg', 'foto1-orb', 2)

imgPanoramic('../img/foto2A.jpg', '../img/foto2B.jpg', 'foto2-sift', 0)
imgPanoramic('../img/foto2A.jpg', '../img/foto2B.jpg', 'foto2-surf', 1)
imgPanoramic('../img/foto2A.jpg', '../img/foto2B.jpg', 'foto2-orb', 2)

imgPanoramic('../img/foto3A.jpg', '../img/foto3B.jpg', 'foto3-sift', 0)
imgPanoramic('../img/foto3A.jpg', '../img/foto3B.jpg', 'foto3-surf', 1)
imgPanoramic('../img/foto3A.jpg', '../img/foto3B.jpg', 'foto3-orb', 2)

imgPanoramic('../img/foto4A.jpg', '../img/foto4B.jpg', 'foto4-sift', 0)
imgPanoramic('../img/foto4A.jpg', '../img/foto4B.jpg', 'foto4-surf', 1)
imgPanoramic('../img/foto4A.jpg', '../img/foto4B.jpg', 'foto4-orb', 2)

# imgPanoramic('../img/foto5A.jpg', '../img/foto5B.jpg', 'foto5-sift', 0)
# imgPanoramic('../img/foto5A.jpg', '../img/foto5B.jpg', 'foto5-surf', 1)
# imgPanoramic('../img/foto5A.jpg', '../img/foto5B.jpg', 'foto5-orb', 2)