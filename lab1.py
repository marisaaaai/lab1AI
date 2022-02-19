'''
Laboratorio 1
Maria Montoya 19169
Maria Morales 19145
'''
from PIL import Image, ImageDraw
from modulos import *
import gc



print("---------------------------Bienvenido a lab1-----------------")
img = input("Ingrese el nombre en conjunto al formato para leer: ")
a = discretizador(img)[0]
start = discretizador(img)[1]
lista_destinos=discretizador(img)[2]
w=0
while (w!=5):
    print("\n---------------------------Menu-----------------")
    print("1. BFS")
    print("2. DFS")
    print("3. Heur")
    print("4. heur")
    print("5. salir")
    w = int(input("Ingrese el numero de la opcion a elegir: "))
    if(w==1):
        paths=[]
        for i in range(len(lista_destinos)):
            paths.append(BFS_paths(a,start,lista_destinos[i]))
        optimal= min(paths)
        print("\nPunto final mas cercano detectado en coordenada: "+str(lista_destinos[paths.index(optimal)]) +" con un numero de pasos de: " + str(optimal))
        
        BFS(a,start,lista_destinos[paths.index(optimal)])
        o = int(input("Desea regresar al menu? \n1.Si\n2.No"))
        if (o ==2):
            w=5
    if(w==2):
        paths=[]
        start_a=start[0]
        start_b=start[1]
        end_a=[]
        end_b=[]
        for i in range(len(lista_destinos)):
            end_a.append(lista_destinos[i][0])
            end_b.append(lista_destinos[i][1])        
        DFS(a,start_a,start_b,end_a[0],end_b[0])
        o = int(input("Desea regresar al menu? \n1.Si\n2.No"))
        if (o ==2):
            w=5
    if(w==3):
        path = aStar(a, start[0], lista_destinos[1])
        
print("Gracias por chequear el programa :)")

        

