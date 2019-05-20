import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../img/bitmap.pbm', 0)
img = img/255
img = 1-img

# Passos (1)-(2): dilatação seguida de erosão com elemento 1x100
kernel1 = np.ones((1, 100), np.uint8)
dilatation1 = cv2.dilate(img, kernel1)
erosion1 = cv2.erode(dilatation1, kernel1)

# Passos (3)-(4): dilatação seguida de erosão com elemento 1x200
kernel2 = np.ones((200, 1), np.uint8)
dilatation2 = cv2.dilate(img, kernel2)
erosion2 = cv2.erode(dilatation2, kernel2)

# Passo (5): Interseção entre (2) e (4)
intersection = cv2.bitwise_and(erosion1, erosion2)

# Passo (6): Fechamento com elemento 1x30
kernel3 = np.ones((1, 30), np.uint8)
closing = cv2.morphologyEx(intersection, cv2.MORPH_CLOSE, kernel3)

# Passo (7): Identificação de componentes conexos

closing = closing.astype(np.uint8)
ret, labels = cv2.connectedComponents(closing)

label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue==0] = 0

print(ret)

fig = plt.figure()
plt.imshow(labeled_img)
plt.show()

for label in range(1,ret):
    mask = np.array(labels, dtype=np.uint8)
    mask[labels == label] = 255

cnts, hierarchy= cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for contour in cnts:
    boxes.append(cv2.boundingRect(contour))

for i in boxes:
    comps = cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(1))
fig = plt.figure()
plt.imshow(1-comps,cmap="gray")
plt.show()

# Passo (8): Para cada componente Conexo

    # Razão entre os pixels preto e o número total de pixels

    # Razão entre transições horizontais e verticais de pixels pretos e brancos e o número total de pixels pretos  

# Passo (9): Classificação dos componentes conexos em texto e não texto

#Passo (10): Seguimentação de linhas e palavras e cálculo das respectivas quantidades

cv2.imwrite("imgRes/01-bitmap-dilated1.pbm", 1-dilatation1)
cv2.imwrite("imgRes/02-bitmap-eroded1.pbm", 1-erosion1)
cv2.imwrite("imgRes/03-bitmap-dilated2.pbm", 1-dilatation2)
cv2.imwrite("imgRes/04-bitmap-eroded2.pbm", 1-erosion2)
cv2.imwrite("imgRes/05-bitmap-intersection.pbm", 1-intersection)
cv2.imwrite("imgRes/06-bitmap-closing.pbm", 1-closing)
cv2.imwrite("imgRes/07-comps.pbm", 1-comps)