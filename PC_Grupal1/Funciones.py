#!/usr/bin/env python
# coding: utf-8

def imshow(img):
    '''
    Función para mostrar una imagen por pantalla con el criterio que considero más acertado.
    
    '''
    from matplotlib import pyplot as plt
    
    
    fig, ax = plt.subplots(figsize=(7, 7))
    # El comando que realmente muestra la imagen
    ax.imshow(img,cmap=plt.cm.gray)
    # Para evitar que aparezcan los números en los ejes
    ax.set_xticks([]), ax.set_yticks([])
    plt.show()


def showImages(img1,titleImg1,img2,titleImg2):
    """
    Función para enseñar en pantalla 2 imagenes
    """
    import cv2
    from matplotlib import pyplot as plt
    
    plt.figure(figsize=(10,8))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)),plt.title(titleImg1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)),plt.title(titleImg2)
    plt.xticks([]), plt.yticks([])
    plt.show()


def identificarCirculo(matriz, numFila, numColumna):
    """
    Esta función la usaremos en la función getNumCircles. Identifica si un pixel tiene alrededor suyo (maximo a 4 pixeles) 
    algun objeto que ya haya sido marcado. Si lo encuentra, devuelve el valor que contiene el objeto, si no, devuelve 0.
    
    """
    import numpy as np
    
    maxVecinos=4
    maxFila,maxColumn=matriz.shape
    for i in range(-maxVecinos,maxVecinos+1):
        if(numFila+i>=maxFila or numFila+i<0 ):
            continue
        for j in range(-maxVecinos,maxVecinos+1):
            if(numColumna+j>=0 or numColumna+j<maxColumn):
                val=matriz[numFila+i][numColumna+j]
                if((val!=0) and (val!=255)):
                    return val    
    return 0


def getNumCircles(matriz):
    """
    Esta función cuenta la cantidad de objetos que se encuentran en una imagen.
    
    """
    import numpy as np

    numFila=0
    numColumna=0
    cont=1
    maxF,maxC=matriz.shape
    for fila in matriz:
        numColumna=0
        for valor in fila:   
            #si el pixel es blanco, buscamos si pertenece a un objeto nuevo o a algun objeto ya encontrado.
            if(matriz[numFila][numColumna]==255):
                val=identificarCirculo(matriz, numFila, numColumna)
                #si no, devuelve 0, al bit le asigna la etiqueca del objeto que se encuentra a su alrededor
                if(val!=0):
                    matriz[numFila][numColumna]=val
                #si no tiene ningun objeto identificado cerca, asigna a ese pixel el valor de un objeto nuevo
                else :
                    matriz[numFila][numColumna]=255-cont
                    cont=cont+1
                        
            numColumna+=1
        numFila+=1
        #devuelve cont-1 porque el contador empieza en 1 no en 0
    return cont-1


def contar(imgTitle,EE_size,EE_type,dilate,mult):
    '''
    Funfción que actúa como filtro para contar objetos de una imagen.
    
    imgTitle:   Nombre de la imagen a la que queremos aplicar el filtro. Debe ir entre '' y con el nombre de la extensión.
                Ejemplo: 'canicas.jpg' o 'balones.png'.
    
    EE_size:    Número de píxeles para el tamaño del elemento estructurador. Dependiendo de EE_type, construiremos el tamaño del
                elemento según sea necesario. Por ejemplo, si el EE es un disco (disk()), simplemente pasaremos EE_size a la
                función. Si es un rectángulo (rectangle()), este tipo de EE necesita como argumento una tupla, constuiremos 
                EE_size=(EE_size,EE_size) y se lo pasaremos como argumento.
                
    EE_type:    Tipo de elemento estructurador.
    
    dilate:     Argumento booleano para saber si aplicar dilatación a la imagen segmentada si tiene ruido. 
                Si dilate==True, se le aplica.
                
    mult:       Si dilatamos la imagen, habrá que usar un tamaño de elemento estructurador un poco más grande que el original
                para erosionar; mult es un  multiplicador para EE_size
    
    '''
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    from skimage import img_as_float
    from skimage.morphology import disk, diamond, ball, rectangle, star, square
    from skimage.morphology import erosion,dilation,opening,closing


    
    # Cargamos la imagen que el usuario pasa como input
    
    try:
        
        img=cv2.imread(imgTitle)
    
    except:
        print('Ha habido un error al cargar la imagen. Compruebe si ha introducido el nombre entre '' y la extensión')
        
    # Enseñamos la imagen original
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title(imgTitle)
    
    # Pasamos a escala de grises y luego segmentamos con Otsu:
    
    img_gris = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, img_th = cv2.threshold(img_gris,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Comprobamos si EE_size tiene el formato adecuado:
    try:
        
        EE_size % 2 == 0
            
    except:
        
        print('ERROR: Introduzca un número impar')
    
    # Comprobamos si la imagen cargada requiere dilatación antes de la erosión, y también el tipo de elemento estructurador:
    try:
        if dilate==True:

            if EE_type == 'disk':
                img_dilation=dilation(img_th,disk(EE_size))
                img_erosion=erosion(img_dilation,disk(EE_size*mult))

            if EE_type == 'diamond':
                img_dilation=dilation(img_th,diamond(EE_size))
                img_erosion=erosion(img_dilation,diamond(EE_size*mult))

            if EE_type == 'rectangle':
                img_dilation=dilation(img_th,rectangle(EE_size,EE_size))
                EE_size=EE_size*mult
                img_erosion=erosion(img_dilation,rectangle(EE_size,EE_size))

            if EE_type == 'ball':
                img_dilation=dilation(img_th,ball(EE_size))
                img_erosion=erosion(img_dilation,ball(EE_size*mult))

            if EE_type == 'star':
                img_dilation=dilation(img_th,star(EE_size))
                img_erosion=erosion(img_dilation,star(EE_size*mult))

            if EE_type == 'square':
                img_dilation=dilation(img_th,square(EE_size))
                img_erosion=erosion(img_dilation,square(EE_size*mult))
        else:

            if EE_type == 'disk':

                img_erosion=erosion(img_th,disk(EE_size))

            if EE_type == 'diamond':
                img_erosion=erosion(img_th,diamond(EE_size))

            if EE_type == 'rectangle':
                img_erosion=erosion(img_th,rectangle(EE_size,EE_size))

            if EE_type == 'ball':
                img_erosion=erosion(img_th,ball(EE_size))

            if EE_type == 'star':
                img_erosion=erosion(img_th,star(EE_size))

            if EE_type == 'square':
                img_erosion=erosion(img_th,square(EE_size))
    except:
        print('Ha habido un error')
        
    cont=getNumCircles(img_erosion.copy())
    
    print("Número de canicas que se encuentran en la imagen: ",cont)
    return cont