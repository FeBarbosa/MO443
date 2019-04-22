from scipy import misc
import matplotlib.pyplot as plt

# A funcao recebe o valor gama a correcao e um nome para salvar a imagem em arquivo
def gammaCorrection(gammaValue, name):
    img = misc.imread('img/baboon.png') # carrega a imagem em uma matriz

    img = img/255 # Normalizacao para [0, 1]

    img = img ** (1/gammaValue) # Aplicacao da correcao gama

    img = img * 255 # Normalizacao para [0, 255]

    

    misc.imsave('img/E1/'+name+'.png', img)
#enddef

gammaCorrection(1.5, 'baboon-a')
gammaCorrection(2.5, 'baboon-b')
gammaCorrection(3.5, 'baboon-c')