'''
Funciones
'''
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import math

def dicretizador():
    image = Image.open('Test2.bmp')
    new_image = image.resize((50, 50))

    temp = np.array(new_image)
    r = 0
    g = 0
    b = 0
    mean = 0
    lista = []
    dif = 0
    nmax = 0
    nmin = 0
    new_size = math.sqrt(temp.size/3)
    new_size = int(new_size)
    a = np.zeros(shape=(new_size,new_size))
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            lista = temp[i][j]
            r = lista[0]
            g = lista[1]
            b = lista[2]
            mean = sum(lista)/3
            nmax = max(lista)
            nmin = min(lista)
            dif = nmax - nmin
            if(dif<15):
                if mean>200:
                    a[i][j] = 0 #el pixel es blanco
                else:
                    a[i][j] = 1 #el pixel es negro
            else:
                if((r>g) and (r>b)):
                    a[i][j] = 2 #el pixel es rojo
                else:
                    if((g>r) and (g>b)):
                        a[i][j] = 3 #el pixel es verde
                    else:
                        a[i][j] = 4 #el pixel es azul
                    

        lista = []
        r = 0
        g = 0
        b = 0
        mean = 0
    with open('outfile.txt','wb') as f:
        np.savetxt(f, a, fmt='%.2f')            

    rojos = np.where(a == 2)
    listOfCoordinates= list(zip(rojos[0], rojos[1]))
    start = []
    for cord in listOfCoordinates:
        start= cord
    verdes = np.where(a == 3)
    listOfCoordinates= list(zip(verdes[0], verdes[1]))
    cantidad_destinos=[]
    for cord in listOfCoordinates:
        cantidad_destinos.append(cord)
    p = np.zeros(shape=(new_size,new_size))
    for i in range(len(temp)):
        for j in range(len(temp[i])):
            lista = temp[i][j]
            r = lista[0]
            g = lista[1]
            b = lista[2]
            mean = sum(lista)/3
            nmax = max(lista)
            nmin = min(lista)
            dif = nmax - nmin
            if(dif<15):
                if mean>200:
                    p[i][j] = 0 #el pixel es blanco
                else:
                    p[i][j] = 1 #el pixel es negro
            else:
                p[i][j] = 0 
                               

        lista = []
        r = 0
        g = 0
        b = 0
        mean = 0
    p = p.astype(int)
    return p
