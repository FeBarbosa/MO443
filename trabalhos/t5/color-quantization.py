import cv2
import numpy as np
from sklearn import cluster

def quantization(path, name, num_colors):
    # (1) Leitura da Imagem
    img = cv2.imread(path)

    width, height, depth = img.shape
    img2 = np.reshape(img, (width * height, depth))

    # (2)-(3) Aplicação do k-means e obtenção dos centros dos grupos e rótulos
    model = cluster.KMeans(n_clusters = num_colors)
    labels = model.fit_predict(img2)
    palette = model.cluster_centers_

    # (4) Reconstrução da imagem com cores reduzidas
    img3 = np.reshape(palette[labels], (width, height, palette.shape[1]))

    cv2.imwrite(name+str(num_colors)+'.png', img3)

quantization('../img/foto2A.jpg', 'foto2A', 2)
quantization('../img/foto2A.jpg', 'foto2A', 4)
quantization('../img/foto2A.jpg', 'foto2A', 8)
quantization('../img/foto2A.jpg', 'foto2A', 16)
quantization('../img/foto2A.jpg', 'foto2A', 32)
quantization('../img/foto2A.jpg', 'foto2A', 64)
quantization('../img/foto2A.jpg', 'foto2A', 128)

quantization('../img/peppers-color.png', 'peppers-color', 2)
quantization('../img/peppers-color.png', 'peppers-color', 4)
quantization('../img/peppers-color.png', 'peppers-color', 8)
quantization('../img/peppers-color.png', 'peppers-color', 16)
quantization('../img/peppers-color.png', 'peppers-color', 32)
quantization('../img/peppers-color.png', 'peppers-color', 64)
quantization('../img/peppers-color.png', 'peppers-color', 128)
