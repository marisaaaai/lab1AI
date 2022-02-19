'''
Laboratorio 1
Maria Montoya 19169
Maria Morales 19145
'''
from PIL import Image, ImageDraw
import numpy as np

import math

zoom = 20
borders = 6
images = []

def discretizador(imagen):
    image = Image.open(imagen)
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
#     with open('outfile.txt','wb') as f:
#         np.savetxt(f, a, fmt='%.2f')            

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
    p= p.astype(int)
    return p,start,cantidad_destinos
        
'''
Codigo extraido de:
https://levelup.gitconnected.com/solve-a-maze-with-python-e9f0580979a1
'''

def BFS_paths(a,start,end):
    def make_step(k):
      for i in range(len(m)):
        for j in range(len(m[i])):
          if m[i][j] == k:
            if i>0 and m[i-1][j] == 0 and a[i-1][j] == 0:
              m[i-1][j] = k + 1
            if j>0 and m[i][j-1] == 0 and a[i][j-1] == 0:
              m[i][j-1] = k + 1
            if i<len(m)-1 and m[i+1][j] == 0 and a[i+1][j] == 0:
              m[i+1][j] = k + 1
            if j<len(m[i])-1 and m[i][j+1] == 0 and a[i][j+1] == 0:
               m[i][j+1] = k + 1

    m = []
    for i in range(len(a)):
        m.append([])
        for j in range(len(a[i])):
            m[-1].append(0)
    i,j = start
    m[i][j] = 1

    k = 0
    while m[end[0]][end[1]] == 0:
        k += 1
        make_step(k)

    i, j = end
    k = m[i][j]
    the_path = [(i,j)]
    while k > 1:
      if i > 0 and m[i - 1][j] == k-1:
        i, j = i-1, j
        the_path.append((i, j))
        k-=1
      elif j > 0 and m[i][j - 1] == k-1:
        i, j = i, j-1
        the_path.append((i, j))
        k-=1
      elif i < len(m) - 1 and m[i + 1][j] == k-1:
        i, j = i+1, j
        the_path.append((i, j))
        k-=1
      elif j < len(m[i]) - 1 and m[i][j + 1] == k-1:
        i, j = i, j+1
        the_path.append((i, j))
        k -= 1

    return len(the_path)
'''
Codigo extraido de:
https://levelup.gitconnected.com/solve-a-maze-with-python-e9f0580979a1
'''
def BFS(a,start,end):
    def make_step(k):
        #Escaneamos la matriz con una doble loop 
      for i in range(len(m)):
        for j in range(len(m[i])):
          if m[i][j] == k:
              #si se encuentra con un numero k chquear si las celdas adyacentes o son no paredes o si no tienen otro numero que 0 o 1
              #si cumple las condiciones le asigna k+1
            if i>0 and m[i-1][j] == 0 and a[i-1][j] == 0:
              m[i-1][j] = k + 1
            if j>0 and m[i][j-1] == 0 and a[i][j-1] == 0:
              m[i][j-1] = k + 1
            if i<len(m)-1 and m[i+1][j] == 0 and a[i+1][j] == 0:
              m[i+1][j] = k + 1
            if j<len(m[i])-1 and m[i][j+1] == 0 and a[i][j+1] == 0:
               m[i][j+1] = k + 1

    def print_m(m):
        for i in range(len(m)):
            for j in range(len(m[i])):
                print( str(m[i][j]).ljust(2),end=' ')
            print()

    def draw_matrix(a,m, the_path = []):
        im = Image.new('RGB', (zoom * len(a[0]), zoom * len(a)), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        for i in range(len(a)):
            for j in range(len(a[i])):
                color = (255, 255, 255)
                r = 0
                if a[i][j] == 1:
                    color = (0, 0, 0)
                if i == start[0] and j == start[1]:
                    color = (0, 255, 0)
                    r = borders
                if i == end[0] and j == end[1]:
                    color = (0, 255, 0)
                    r = borders
                draw.rectangle((j*zoom+r, i*zoom+r, j*zoom+zoom-r-1, i*zoom+zoom-r-1), fill=color)
                if m[i][j] > 0:
                    r = borders
                    draw.ellipse((j * zoom + r, i * zoom + r, j * zoom + zoom - r - 1, i * zoom + zoom - r - 1),
                                   fill=(0,255,0))
        for u in range(len(the_path)-1):
            y = the_path[u][0]*zoom + int(zoom/2)
            x = the_path[u][1]*zoom + int(zoom/2)
            y1 = the_path[u+1][0]*zoom + int(zoom/2)
            x1 = the_path[u+1][1]*zoom + int(zoom/2)
            draw.line((x,y,x1,y1), fill=(0, 255,0), width=5)
        draw.rectangle((0, 0, zoom * len(a[0]), zoom * len(a)), outline=(0,255,0), width=2)
        images.append(im)


    m = []
    for i in range(len(a)):
        m.append([])
        for j in range(len(a[i])):
            m[-1].append(0)
    i,j = start
    m[i][j] = 1

    k = 0
    while m[end[0]][end[1]] == 0:
        k += 1
        make_step(k)
        draw_matrix(a, m)

    #ir al punto final
        #buscar k-1 ir a esa celda y quitarle uno a k
        #repetir hasta llegar a donde k=1
        #guaardar las celdas a the_path
    i, j = end
    k = m[i][j]
    the_path = [(i,j)]
    while k > 1:
      if i > 0 and m[i - 1][j] == k-1:
        i, j = i-1, j
        the_path.append((i, j))
        k-=1
      elif j > 0 and m[i][j - 1] == k-1:
        i, j = i, j-1
        the_path.append((i, j))
        k-=1
      elif i < len(m) - 1 and m[i + 1][j] == k-1:
        i, j = i+1, j
        the_path.append((i, j))
        k-=1
      elif j < len(m[i]) - 1 and m[i][j + 1] == k-1:
        i, j = i, j+1
        the_path.append((i, j))
        k -= 1
      draw_matrix(a, m, the_path)

    for i in range(10):
        if i % 2 == 0:
            draw_matrix(a, m, the_path)
        else:
            draw_matrix(a, m)

    #print_m(m)

    im= images[len(images)-2]
    im.show()
'''
Codigo extraido de:
https://levelup.gitconnected.com/solve-a-maze-with-python-e9f0580979a1
'''
def DFS(p,start_a,start_b,end_a,end_b):
    images = []
    lenghts=[]
    a=p
    start_i=start_a
    start_j=start_b
    end_i=end_a
    end_j=end_b

    path_so_far = []
    #print("prueba de que funciona")
    lent=0
    ##########################################################
    
    def draw_matrix(a, the_path=[]):
        im = Image.new('RGB', (zoom * len(a[0]), zoom * len(a)), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        for i in range(len(a)):
            for j in range(len(a[i])):
                color = (255, 255, 255)
                r = 0
                if a[i][j] == 1:
                    color = (0, 0, 0)
                if i == start_j and j == start_j:
                    color = (0, 255, 0)
                    r = borders
                if i == end_i and j == end_j:
                    color = (0, 255, 0)
                    r = borders
                draw.rectangle((j*zoom+r, i*zoom+r, j*zoom+zoom-r-1, i*zoom+zoom-r-1), fill=color)
                if a[i][j] == 2:
                    r = borders
                    draw.ellipse((j * zoom + r, i * zoom + r, j * zoom + zoom - r - 1, i * zoom + zoom - r - 1),
                                   fill=(128, 128, 128))
        for u in range(len(the_path)-1):
            y = the_path[u][0]*zoom + int(zoom/2)
            x = the_path[u][1]*zoom + int(zoom/2)
            y1 = the_path[u+1][0]*zoom + int(zoom/2)
            x1 = the_path[u+1][1]*zoom + int(zoom/2)
            draw.line((x,y,x1,y1), fill=(0, 255, 0), width=5)
        draw.rectangle((0, 0, zoom * len(a[0]), zoom * len(a)), outline=(0, 255, 0), width=2)
        images.append(im)
        
    
    
    ########################################################
    def go_to(i, j):
        
        if i < 0 or j < 0 or i > len(a)-1 or j > len(a[0])-1:
            
            return 
        # If we've already been there or there is a wall, quit
        if (i, j) in path_so_far or a[i][j] > 0:
            
            return 
        path_so_far.append((i, j))
        a[i][j] = 2
        #draw_matrix(a, path_so_far)
        if (i, j) == (end_i, end_j):
            lent = len(path_so_far)
            
            #print(path_so_far)
            for animate in range(10):
                if animate % 2 == 0:
                    draw_matrix(a, path_so_far)
                else:
                    draw_matrix(a)
            lenghts.append(len(path_so_far))
            path_so_far.pop()
            
            return
            
            
        else:
            go_to(i - 1, j)  # check top
            go_to(i + 1, j)  # check bottom
            go_to(i, j + 1)  # check right
            go_to(i, j - 1)  # check left
        path_so_far.pop()
        #draw_matrix(a, path_so_far)
        return 


    go_to(start_i, start_j)
    im= images[len(images)-2]
    im.show()

#Calcular la distancia manhattan  
def h(cell1, cell2):
    x1,y1 = cell1
    x2,y2 = cell2
    
    return abs(x1-x2) + abs(y1-y2)

'''
Codigo extraido de:
https://levelup.gitconnected.com/an-animated-guide-to-implementing-dfs-in-python-to-find-a-path-1e939b7b8599
'''
#Recibe la matriz 
def aStar(m, start, end):
    
    maze = []
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    maze.append([" "," ", " ", " ", " ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "," "," ", " "," "," "])
    
    empty = []
    for i in range (1,26):
        empty.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    
    for i in range(len(maze)):
        for j in range(len(maze)):
            if m[i][j][0] == 0 and m[i][j][1] == 0:
                empty[i][j] = 1
                
        
    dictionary={}
    grid=[]
    g_score={cell:float('inf') for cell in grid}
    g_score[start]=0
    f_score={cell:float('inf') for cell in grid}
    f_score[start]=h(start,(1,1))
    
    open=PriorityQueue()
    open.put((h(start,(1,1)),h(start,(1,1)),start))
    aPath={}
        

    for i in range (1,26):
        for j in range(1,26):
            grid.append((j,i))

    for i in grid:
        dictionary[i]= {'E':0,'W': 0, 'N': 0, 'S': 0}
        
    for x in grid:
        i,j=x
        if empty[i-1][j-1]==0:
            if i==1 and j==1:
                if empty[i-1][j]==0:
                    dictionary[(i,j)]['W']=1
                if empty[i][j-1]==0:
                    dictionary[(i,j)]['S']=1
            elif i==20 and j==20:
                if empty[i-2][j-1]==0:
                    dictionary[(i,j)]['N']=1
                if empty[i-1][j-2]==0:
                    dictionary[(i,j)]['E']=1
            elif i==1 and j==20:
                if empty[i][j-1]==0:
                    dictionary[(i,j)]['W']=1
                if empty[i-1][j-2]==0:
                    dictionary[(i,j)]['S']=1
            elif i==20 and j==1:
                if empty[i-2][j-1]==0:
                    dictionary[(i,j)]['N']=1
                if empty[i-1][j]==0:
                    dictionary[(i,j)]['E']=1
            elif j==1:
                if empty[i-2][j-1]==0:
                    dictionary[(i,j)]['N']=1
                if empty[i][j-1]==0:
                    dictionary[(i,j)]['S']=1
                if empty[i-1][j]==0:
                    dictionary[(i,j)]['E']=1
            elif j==20:
                if empty[i][j-1]==0:
                    dictionary[(i,j)]['S']=1
                if empty[i-1][j-2]==0:
                    dictionary[(i,j)]['W']=1
                if empty[i-2][j-1]==0:
                    dictionary[(i,j)]['N']=1
            elif i==1:
                if empty[i-1][j-2]==0:
                    dictionary[(i,j)]['W']=1
                if empty[i-1][j]==0:
                    dictionary[(i,j)]['E']=1
                if empty[i][j-1]==0:
                    dictionary[(i,j)]['S']=1
            elif i==20:
                if empty[i-1][j-2]==0:
                    dictionary[(i,j)]['W']=1
                if empty[i-1][j]==0:
                    dictionary[(i,j)]['E']=1
                if empty[i-2][j-1]==0:
                    dictionary[(i,j)]['N']=1
            else:
                if empty[i-1][j-2]==0:
                    dictionary[(i,j)]['W']=1
                if empty[i-1][j]==0:
                    dictionary[(i,j)]['E']=1
                if empty[i][j-1]==0:
                    dictionary[(i,j)]['S']=1
                if empty[i-2][j-1]==0:
                    dictionary[(i,j)]['N']=1
    while not open.empty():
        currCell=open.get()[2]
        if currCell==(end1):
            break
        for d in 'ESNW':
            if dictionary[currCell][d]==True:
                if d=='E':
                    childCell=(currCell[0],currCell[1]+1)
                if d=='W':
                    childCell=(currCell[0],currCell[1]-1)
                if d=='N':
                    childCell=(currCell[0]-1,currCell[1])
                if d=='S':
                    childCell=(currCell[0]+1,currCell[1])

                temp_g_score=g_score[currCell]+1
                temp_f_score=temp_g_score+h(childCell,(1,1))

                if temp_f_score < f_score[childCell]:
                    g_score[childCell]= temp_g_score
                    f_score[childCell]= temp_f_score
                    open.put((temp_f_score,h(childCell,(1,1)),childCell))
                    aPath[childCell]=currCell
    fwdPath={}
    cell=(end)
    
    while cell!=start:
        fwdPath[aPath[cell]]=cell
        cell=aPath[cell]
        
    return fwdPath 
    