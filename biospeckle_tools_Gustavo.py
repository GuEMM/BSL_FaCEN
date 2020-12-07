# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:07:14 2020

@author: Gustavo Eduardo MEreles Menesse - FaCEN-UNA
"""
from numpy import *

def datapack(DIR=None,ni=1,nf=[]):
    ''' Import data from .bmp images in a directory to a 3D numpy array (t,x,y).
    
    Parameters:
        Dir = .bmp files directory. The files must me name with integers. Example: 0.bmp, 1.bmp, 500.bmp.
            So the function could orginize in ascendent order. If Dir = None, current execution directory will be used.
        ni = Index number of initial .bmp to be packed. Defaul is the first (0) .bmp in the directory.
        nf = Index number of final .bmp to be packed.
    '''
    import numpy as np
    import cv2, os
    
    if DIR!=None:
        allfiles=os.listdir(os.getcwd()+'\\'+ DIR)

    else:
        allfiles=os.listdir(os.getcwd())
        print('We will work with the images in',os.getcwd())

    #Crea una lista con todos los archivos de allfiles con extension bmp
    oimlist=[filename for filename in allfiles if  filename[-4:] in [".bmp",".BMP"]]

    #Ordenar las imagenes de manera ascendente
    oimlist=sorted(oimlist,key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    if DIR!='':
        DIR=DIR+'\\'
        arr = np.array(cv2.imread( DIR + oimlist[1],cv2.IMREAD_GRAYSCALE), dtype=np.float)
    else:
        arr = np.array(cv2.imread(oimlist[1],cv2.IMREAD_GRAYSCALE), dtype=np.float)
    
    f,c=arr.shape
    
    if nf==[]:
        nf = len(oimlist)
    
    N = int(nf-ni)
    
    DATA=np.zeros((N,f,c))
    
    for i in range(0,N,1):
        
        if DIR!='':
            DATA[i,:,:] = np.array(cv2.imread(DIR + '\\' + oimlist[i+ni],cv2.IMREAD_GRAYSCALE), dtype=np.float)
        else:
            DATA[i,:,:] = np.array(cv2.imread(oimlist[i+ni],cv2.IMREAD_GRAYSCALE), dtype=np.float)
    
    return DATA

def clean_all_bmp(DIR=None):
    '''Delete all .bmp files in a directory'''
    import os
    
    if DIR!=None:
        allfiles=os.listdir(os.getcwd()+'\\'+ DIR)

    else:
        allfiles=os.listdir(os.getcwd())
        print('We will work with the images in',os.getcwd())
    
    reslist=[filename for filename in allfiles if  filename[-4:] in [".bmp",".BMP"]]

    #Delete all remainder images if they exist
    if reslist!=[]:
        for i in range(0,len(reslist)):
            nm = reslist[i]
            os.remove(nm)
    return

def selection(Data,DIR,oimlist):
    '''Select a region of an image'''
    import os, cv2

#####################################################
    #Abre y carga las imagenes
    im = cv2.imread(os.getcwd()+'\\'+ DIR + '\\' + oimlist[1])

    #####################################################
    print('Initializing //')

    ###Nota#################################################
    # Si la resolucion de las imagenes es muy alta, no cabe en pantalla al visualizarlas en una ventana
    # por lo que aquí se realiza un ajuste de las dimensiones
    ########################################################
    #Toma la resolución de la primera imagen de la lista
    w,h = Data.shape
    
    print(w,h)
    #Reajustar imagen si la resolucion es muy grande y no entrara en pantalla
    if w>1500:        
        f=4
    elif w>1000:
        f=2
    else:
        f=1

    res=[int(w/f),int(h/f)]

    # Crea una dupla con la resolución reajustada
    res=[w/f,h/f]

    #Redefine el tamaño de la imagen para mostrar posteriormente en pantalla y solicitar la seleccion de la region de interes
    imS = cv2.resize(im, (int(res[1]), int(res[0]))) 

######################################################
#SELECCION DE LA REGION DE INTERES
######################################################
    print('\n')
    print('Select a region of interest, confirm it pressing enter twice')

# Solicita la selección de la región de interes
    r = cv2.selectROI(imS)

#Redefine la región seleccionada para las resolución original de las imágenes
    r=[f*r[0],f*r[1],f*r[2],f*r[3]]

#Cierran las ventanas donde se abrio la ROI selector
    cv2.waitKey(0)
    cv2.destroyAllWindows()
     
    return r

def crop_and_save(Data,r,DIR,Dir,ni=0):
        import numpy as np
        import os,cv2

        DataCrop = np.zeros([Data.shape[0],int(r[1]+r[3])-int(r[1]),int(r[0]+r[2])-int(r[0])])

        for i in range(Data.shape[0]):
        #Toma la imagen en la region de interes
            DataCrop[i] = Data[i][int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        #####################################################
        #Reconstrucción del video en la región de interés
        #####################################################
        #Toma las dimensiones de una imagen cortada
        height, width = DataCrop[0].shape

        if Dir=='':
            direc = DIR + '_c'  #+Vname.split('.'+Format)[0]
        
        else:
            direc = Dir

        if not os.path.exists(direc):
            os.makedirs(direc)

    #Prepara el modulo de escritura de video con el mismo fps y la nueva resolucion size 

    #Realiza la escritura de todas las imagenes cortadas contenidas en la lista imgc_array
        for i in range(DataCrop.shape[0]):
            name = direc +'\\'+ str(ni+i) + '.bmp'
            cv2.imwrite(name,DataCrop[i])

            print('#', end='')

        print('The cropped images was saved in'+ direc )

        return 
    

def crop_Im(DIR='', Dir='', r = []):
    '''Crop in some Region of Interest all existent images in a directory and save it in a _c directory'''
        
    import os
    
    nn = 500
    
    if DIR!='':
        
        allfiles=os.listdir(os.getcwd()+'\\'+ DIR)

    else:
        allfiles=os.listdir(os.getcwd())
        print('We will work with the images in',os.getcwd())

    #Crea una lista con todos los archivos de allfiles con extension bmp
    oimlist=[filename for filename in allfiles if  filename[-4:] in [".bmp",".BMP"]]

    #Ordenar las imagenes de manera ascendente
    oimlist=sorted(oimlist,key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    if len(oimlist)<nn:
    
        Data = datapack(DIR)

        if r==[]:

            r = selection(Data[5],DIR,oimlist)
        
        if len(r)==4:
               
            crop_and_save(Data,r,DIR,Dir)
        
        else:
               print('I need more information about de ROI. Abort cropping!')
    else:
        n = len(oimlist)//nn
        
        Data = datapack(DIR,ni=3,nf=4)

        if r==[]:

            r = selection(Data[0],DIR,oimlist)
        
        for i in range(n):
            
            Data = datapack(DIR,ni=nn*i,nf=nn*(1+i))
            
            if r==[]:

                r = selection(Data[5],DIR,oimlist)

            if len(r)==4:

                crop_and_save(Data,r,DIR,Dir,ni=nn*i)

            else:
                   print('I need more information about de ROI. Abort cropping!')
                    
        if len(oimlist)%nn!=0:
            
            Data = datapack(DIR,ni=n*nn,nf= n*nn+len(oimlist)%nn)
            
            if r==[]:

                r = selection(Data[5],DIR,oimlist)

            if len(r)==4:

                crop_and_save(Data,r,DIR,Dir,ni=n*nn)

            else:
                   print('I need more information about de ROI. Abort cropping!')
        
    return r

def CropAll():

    import os 

    allfiles = os.listdir(os.getcwd())

    Dirs = []

    for i in allfiles:
        Dir = os.path.isdir(i)

        if Dir == True:
            if str('.') not in i:
                Dirs.append(i)

    print('Carpetas disponibles')

    for i in range(0,len(Dirs)):
        print(i,'-',Dirs[i])

    n = int(input('Elija una carpeta ingresando su numero (int):'))

    allfiles=os.listdir(Dirs[n])

    files = []

    for i in allfiles:
            Dir = os.path.isdir(i)

            if Dir != True:
                if str('.') not in i:
                        files.append(i)
    
    for i in range(len(files)):

        DIR = Dirs[n] + '//' + files[i]

        Dir = Dirs[n] + '_c//' + files[i]
        
        print(DIR)

        if i==0:

            r = crop_Im(DIR,Dir)
        else:
            crop_Im(DIR,Dir,r)
    return


def VideoWritter(Dir='',forma='png', fps=10, clean=False):
    '''Create a video from a set of images on a directory.
    
    Parameters
        -Dir: Directory where you keep the image set. Default: execution directory
        -forma: extension of the images, example: bmp,png,jpeg,etc. Default: png 
        -fps: define the frames per second of video. Default: 10fps
        -clean: deletes all images after finish the video
    '''
    import os, cv2

    if Dir=='':

        allfiles=os.listdir(os.getcwd())

    else:
        allfiles=os.listdir(Dir)


    Imlist=[filename for filename in allfiles if  filename[-4:] in ["."+forma]]

    if Imlist==[]:
        print('There is no bmp images in this directory. Please, check it!')

    else:
        Imlist=sorted(Imlist,key=lambda f: int(''.join(filter(str.isdigit, f))))

        img_array = []

        for filename in Imlist:
            img = cv2.imread(Dir+'\\'+filename)

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            height, width, l = img.shape
            
            size = (width,height)
            img_array.append(img)

        name=input('Enter a name for video save: ')
        name=name+'.avi'

        video = cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        #Realiza la escritura de todas las imagenes cortadas contenidas en la lista imgc_array
        for i in range(0,len(img_array)):
            video.write(img_array[i])
            print('#', end='')

        video.release()
    
    if clean!=False:
        if Imlist!=[]:
            for i in range(0,len(Imlist)):
                nm = Imlist[i]
                os.remove(nm)
    
    return


def decompose_video(aviname,Format='avi'):
    '''Decompose videos in all its frames, have the option to take averages of frames'''
    import numpy as np
    import cv2, os
    
    direc = aviname.split('.'+Format)[0]
            
    if not os.path.exists(direc):
        os.makedirs(direc)
                
    cap= cv2.VideoCapture(aviname)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total=0
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        total+=1
    cap.release()
    print('Some video properties')
    print('FPS: ',fps)
    print('Total frames number= ',total)
    
    i=0
    
    Med=input('Did you want take an average of groups of n-frames?(y/n):')
    
    while (Med !='y' and Med!='n'):
        Med=input('Please! Answer with y or n. Did you want take an average of groups of n-frames?(y/n):')
    
    if Med == 'y':
        n = int(input('Numbers of images per group (Integer number greater than 1) = '))

        while n < 1:
                print('Please, the number must be an integer number equal or greater than 1')
                n = int(input('Numbers of images per group (Integer number equal or greater than 1)='))
    
    if Med == 'n':
        n=1 #Will not take averages of frames from video
        
    #Initialize counter
    i=0
    j=0
    cap= cv2.VideoCapture(aviname)
    
    while(cap.isOpened()):
        
        if i==n:
            average_img= Sum/n
            
            nam = aviname.split('.'+Format)
            DIR = nam[0]+'\\'+str(j)+'.bmp'  
            
            cv2.imwrite(DIR,average_img)
            cv2.imshow('FRAME', average_img)
            
            i=0
            j+=1
        
        ret, frame = cap.read()
        if ret == False:
            break
        
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        s = gframe.shape
        
        if i==0:
            Sum = np.zeros(s)
        
        if i<n:
            Sum += gframe
            i+=1
        
    cap.release()
    cv2.destroyAllWindows()
    return

def crop_Vid(Vname,Format):
    '''Crop a Video in some Region of Interest and save it as a new video'''
    
    import os
    
    #nombre = input('Enter a name for the new video: ')
    
    # Añade la extesión correspondiente
    nombre = Vname.split('.')[0] + '.' + Format 

    #####################################################
    #Abre el archivo de video
    cap= cv2.VideoCapture(Vname)

    #Obtener el fps del video
    fps = cap.get(cv2.CAP_PROP_FPS)

    #Inicializacion de variables, define dos listas donde se cargaran las imagenes, en una las originales y en la otra la cortada
    img_array=[]
    imgc_array=[]
#####################################################
    print('Initializing //')
    
    #Mientras existan frames nuevos en el video se repetirá el ciclo while
    while(cap.isOpened()):
        # ret(Si se pudo obtener un frame es TRUE, caso contrario es FALSE)
        # En 'frame' se carga la imagen
        ret, frame = cap.read()
    
        if ret == False:
            break
    
        # Convierte la imagen de BGR a escala de grises
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Guarda el frame como imagen bmp
        cv2.imwrite('im.bmp',im) 
    
        # Lee la imagen guardada
        im = cv2.imread('im.bmp')
    
        # Carga la imagen en la lista de imagenes originales
        img_array.append(im)
    
        # Elimina la imagen .bmp del frame
        os.remove('im.bmp')

        print('#',end='')

    #Cierra el modulo de captura de frames
    cap.release()

######################################################

#Nota#################################################
# Si la resolucion de las imagenes es muy alta, no cabe en pantalla al visualizarlas en una ventana
# por lo que aquí se realiza un ajuste de las dimensiones
######################################################

#Toma la resolución de la primera imagen de la lista
    w,h,l = img_array[0].shape

#Reajustar imagen si la resolucion es muy grande y no entrara en pantalla
    if w>1500:        
        f=4
    elif w>1000:
        f=2
    else:
        f=1

    res=[int(w/f),int(h/f)]

# Crea una dupla con la resolución reajustada
    res=[w/f,h/f]

#Redefine el tamaño de la imagen para mostrar posteriormente en pantalla y solucitar la seleccion de la region de interes
    imS = cv2.resize(im, (int(res[1]), int(res[0])))  

######################################################
#SELECCION DE LA REGION DE INTERES
######################################################
    print('\n')
    print('Select a region of interest, confirm it pressing enter twice')

# Solicita la selección de la región de interes
    r = cv2.selectROI(imS)

#Redefine la región seleccionada para las resolución original de las imágenes
    r=[f*r[0],f*r[1],f*r[2],f*r[3]]

#Cierran las ventanas donde se abrio la ROI selector
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#####################################################
#Recorte de imagenes en la Region de Interes
#####################################################

    for i in range(0,len(img_array)):
    #Corta la i-esima imagen
        imCrop = img_array[i][int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    #Adjunta la imagen cortada a la lista 
        imgc_array.append(imCrop)

#####################################################
#Reconstrucción del video en la región de interes
#####################################################
#Toma las dimensiones de una imagen cortada
    height, width, layer = imgc_array[0].shape
    size = (width,height)
    direc = os.getcwd()+'\\' #+Vname.split('.'+Format)[0]
    
    if not os.path.exists(direc):
        os.makedirs(direc)
    direc = direc + nombre
#Prepara el modulo de escritura de video con el mismo fps y la nueva resolucion size 
    video = cv2.VideoWriter(direc,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
#Realiza la escritura de todas las imagenes cortadas contenidas en la lista imgc_array
    for i in range(0,len(imgc_array)):
        video.write(imgc_array[i])
        print('#', end='')

#Cierra el modulo de escritura de video
    video.release()
    cv2.destroyAllWindows()
    print('The new video was saved in'+ direc )
    
    return direc

def time_expousure_reconfig(data,Ftau):
    '''Grouping frames creating a new data set with tau time expousure for MESI (Multi-expousure speckle imaging)) implementation
    
    Parameters:
    -data: Original dataset with dt time expousure
    -Ftau: number of frame for each frames integration block, it must be an integer
    '''
    import numpy as np
    
    Ftau = int(Ftau)
    
    Fn = data.shape[0]//Ftau
    
    DAT = np.zeros([Fn,data.shape[1],data.shape[2]])
    
    for fn in range(Fn):
        
        DAT[fn] = np.int64(np.sum(data[fn*Ftau:(fn+1)*Ftau],axis=0)/Ftau)
    
    return DAT

def Select_frame_from_data(data,Finte):
    '''Take frames from some interval (Ftau) and return a new data set with this frames
    
    Parameter:
    -data: Original dataset with dt time expousure
    -Ftau: Interval to take frames
    '''
    import numpy as np

    Finte = int(Finte)
    
    Fn = data.shape[0]//Finte
    
    DAT = np.zeros([Fn,data.shape[1],data.shape[2]])

    for fn in range(Fn):
        DAT[fn] = data[fn*Finte]
    
    return DAT

def WaveletDecomposition(Data,dt=1,Dir=''):
    '''Wavelet transform for frequency decomposition of speckle dataset using Morlet wavelet.

    Parameters:
    Data: dataset to decompose
    dt: Sampled time interval
    Fint: Frequency interval that will be used to decomposition
    Dir: Save directory, as default will save in a folder name Wavelet_decomposition in the execution directory 
    '''
    from transform import WaveletTransform
    #from mlpy_wavelet import cwt
    import time
    import numpy as np
    
    WaveletAnalysis = WaveletTransform
    
    wa = WaveletAnalysis(Data, dt=dt, unbias=True, axis = 0)
    
    s = wa.scales
    
    FP = wa.fourier_frequencies
    
    print(len(s))
    
    SCALlist = []
    
    Freqlist = []
    
    Fint = int(input('Frequency band interval for reconstruction: '))
    
    #For morlet w0=6
    w0 = 6
     
    DomT = np.max(FP)
        
    N = int(DomT/Fint)+1
    
    for n in range(N):

        sp = np.where((FP>=n*Fint)&(FP<(n+1)*Fint))[0]

        if len(sp)>0:

                SCALlist.append(s[sp])

                Freqlist.append(1/(4 *np.pi*s[sp]/(w0 + (2 + w0 ** 2)**.5)))
    
    """
    NN = int(input('Enter de number of scales per band (int)'))
      
    bands = int(len(s)/NN)+1

    if len(s)%NN!=0:
        print('Last band will have just', len(s)%NN ,'scales elements')
    
    for i in range(bands):

        SCALlist.append(s[NN*i:NN*i+NN])
        
        Freqlist.append(1/(4 *np.pi*s[NN*i:NN*i+NN]/(w0 + (2 + w0 ** 2)**.5)))
    """
    
    return Freqlist, SCALlist

def CWT(Data,dt):
    import numpy as np
    from transform import WaveletTransform
    #from mlpy_wavelet import cwt
    import time

    WaveletAnalysis = WaveletTransform
    
    wa = WaveletAnalysis(Data, dt=dt, unbias=True, axis = 0)
    
    s = wa.scales

    cw = cwt(Data,dt=dt,scales=s,wf='morlet',p=6)

    return cw,s

def Reconstruction(cw, sig, dt, N0, SCALlist, slist=[]):
    import numpy as np
    
    dj = 0.125

    C_d = 0.776

    Y_00 = np.pi**-0.25

    SIGreco = np.zeros([sig.shape[0],sig.shape[1],sig.shape[2]])
    
    #   for b in range(len(SCALlist)):
    if slist!=[]:
        
         SIGreco = (dj * dt ** .5 / (C_d * Y_00))*icwt(cw[slist], dt, SCALlist, wf='morlet', p=2)+sig.mean(keepdims=True)

    else:
        N = SCALlist.shape[0]

        SIGreco = (dj * dt ** .5 / (C_d * Y_00))*icwt(cw[N0:N0+N], dt, SCALlist, wf='morlet', p=2)+sig.mean(keepdims=True)

            #print('#',end='')
                
    return SIGreco

def Wavelet_reconstruct_with_saving(cw,sig,dt,Freqlist,SCALlist,slist=[],Dir=''):
    
    import cv2, os
    import numpy as np
    import PIL
    from PIL import Image
    
    for i in range(len(SCALlist)):

        Freqlims =np.round(Freqlist[i].min(),3),np.round(Freqlist[i].max(),3)

        if Dir!='':

            name = os.getcwd()+'\\Wavelet_decomposition\\'+Dir + '\\' + str(Freqlims[0]) +'_' + str(Freqlims[1]) + 'Hz\\'

            if not os.path.exists(name):

                os.makedirs(name)
        else:
            name = os.getcwd()+'\\Wavelet_decomposition\\Some_dataset\\' + '\\' + str(Freqlims[0]) +'_' + str(Freqlims[1]) + 'Hz\\'

            if not os.path.exists(name):

                os.makedirs(name)

        Xband = Reconstruction(cw,sig,dt,i*SCALlist[i].shape[0],SCALlist[i],slist=slist[i])
        
        EXTR = np.mean(Xband)+13*np.std(Xband)

        Xband[Xband>EXTR]=0

        print('Reconstrucion on Band {}'.format(int(i)),' completed')

        width, height = Xband[0].shape

        size = (width,height)

        for j in range(Xband.shape[0]):

            X = Xband[j]

            IM = np.array(X,dtype=np.uint8)

            Im = Image.fromarray(IM)

            Im.save(name + str(j)+'.bmp')
    
    return 

def Blocks_data(data,wx=10,wy=10):
    '''Divide dataset into many domains to enhance the wavelet transform performance
    
    Parameters:
    data: dataset (time,x,y)
    wx,wy: window dimension
    '''
    import numpy as np
    
    Nx,Ny = data.shape[1]//wx, data.shape[2]//wy
    
    DATA = np.zeros([Nx,Ny,data.shape[0],wx,wy])
    
    for i in range(Nx):
    
        for j in range(Ny):
        
            DATA[i,j,:,:,:] = data[:,i*wx:(i+1)*wx,j*wy:(j+1)*wy]
            
    return DATA

def Join_blocks(Data):
    
    import copy
    import numpy as np
    
    Nx = Data.shape[0]
    
    Ny = Data.shape[1]
    
    D1,D2,H1,H2 = [],[],[],[]
    
    for nx in range(Nx):
        
        for ny in range(Ny-1):
            
            if ny==0:
                
                D1 = np.concatenate([Data[nx,ny],Data[nx,ny+1]],axis=2) 
                
                D2 = D1
                
            else:
                
                D2 = np.concatenate([D1,Data[nx,ny+1]],axis=2) 

                D1 = copy.deepcopy(D2)
                
        if nx==0:

            H1 = copy.deepcopy(D2)
            
        else:
            
            H2 = np.concatenate([H1,D2],axis=1)
            
            H1 = copy.deepcopy(H2)
            
    return H2

def view_blocks(DATABLOCKS):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig,ax = plt.subplots(DATABLOCKS.shape[0],DATABLOCKS.shape[1])

    for i in range(DATABLOCKS.shape[0]):

        for j in range(DATABLOCKS.shape[1]):

            ax[i,j].imshow(DATABLOCKS[i,j,0])

            ax[i,j].axis('off')

            fig.subplots_adjust( # the top of the subplots of the figure
            wspace = 0.1,  # the amount of width reserved for space between subplots,
                          # expressed as a fraction of the average axis width
            hspace = 0.1,  # the amount of height reserved for space between subplots,
                          # expressed as a fraction of the average axis height
            )
            
def Wavelet_Blocks(data,dt=1,wx=25,wy=25,Dir=''):
    
    import os
    import PIL
    from PIL import Image
    from transform import WaveletTransform
    import numpy as np
    
    WaveletAnalysis = WaveletTransform
    
    NN = 2**int(np.log2(data.shape[0]))
    
    data = data[:NN]
    
    DATABLOCKS = Blocks_data(data,wx=wx,wy=wy)
    
    Nx = DATABLOCKS.shape[0]
    Ny = DATABLOCKS.shape[1]
    
    freq, scal = WaveletDecomposition(DATABLOCKS[0,0],dt=dt)
 
    DatablocksWave = np.zeros(DATABLOCKS.shape)
    
    for l in range(len(scal)):
        
        for i in range(Nx):
            
            for j in range(Ny):
                
                print('#',end='')
                
                #cw = WaveletAnalysis(DATABLOCKS[i,j], dt=dt, unbias=True, axis = 0)
                cw = CWT(DATABLOCKS[i,j],dt=dt)[0]
                
                DatablocksWave[i,j] = np.real(Reconstruction(cw,DATABLOCKS[i,j],dt,l*scal[l].shape[0],scal[l]))
        
        dataF = Join_blocks(DatablocksWave)

        print('Reconstrucion on Band {}'.format(int(l)),' completed')
        
        #print(dataF.shape)
        
        Freqlims =np.round(freq[l].min(),3),np.round(freq[l].max(),3)

        if Dir!='':

            name = os.getcwd()+'\\Wavelet_decomposition\\'+Dir + '\\' + str(Freqlims[0]) +'_' + str(Freqlims[1]) + 'Hz\\'

            if not os.path.exists(name):

                os.makedirs(name)
        else:
            name = os.getcwd()+'\\Wavelet_decomposition\\Some_dataset\\' + '\\' + str(Freqlims[0]) +'_' + str(Freqlims[1]) + 'Hz\\'

            if not os.path.exists(name):

                os.makedirs(name)
        
        t, width, height = dataF.shape

        size = (width,height)
        
        EXTR = np.mean(dataF)+13*np.std(dataF)

        dataF[dataF>EXTR]=0
        
        for n in range(dataF.shape[0]):

            X = dataF[n]

            IM = np.array(X,dtype=np.uint8)

            Im = Image.fromarray(IM)

            Im.save(name + str(n)+'.bmp')
    
    print('FINISHED!')
    
    return freq

def normalization(s, dt):
    import numpy as np
    
    PI2 = 2 * np.pi
    
    return np.sqrt((PI2 * s) / dt)


def morletft(s, w, w0, dt):
    """Fourier tranformed morlet function.
    
    Input
      * *s*    - scales
      * *w*    - angular frequencies
      * *w0*   - omega0 (frequency)
      * *dt*   - time step
    Output
      * (normalized) fourier transformed morlet function
    """
    import numpy as np
    
    p = 0.75112554446494251 # pi**(-1.0/4.0)
    wavelet = np.zeros((s.shape[0], w.shape[0]))
    pos = w > 0

    for i in range(s.shape[0]):
        n = normalization(s[i], dt)
        wavelet[i][pos] = n * p * np.exp(-(s[i] * w[pos] - w0)**2 / 2.0)
        
    return wavelet

def angularfreq(N, dt):
    """Compute angular frequencies.
    :Parameters:   
       N : integer
          number of data samples
       dt : float
          time step
    
    :Returns:
        angular frequencies : 1d numpy array
    """
    import numpy as np
    # See (5) at page 64.
    
    N2 = N / 2.0
    w =np.empty(N)

    for i in range(w.shape[0]):
        if i <= N2:
            w[i] = (2 * np.pi * i) / (N * dt)
        else:
            w[i] = (2 * np.pi * (i - N)) / (N * dt)

    return w

def cwt(x, dt, scales, wf='dog', p=2):
    """Continuous Wavelet Tranform.
    :Parameters:
       x : 1d array_like object
          data
       dt : float
          time step
       scales : 1d array_like object
          scales
       wf : string ('morlet', 'paul', 'dog')
          wavelet function
       p : float
          wavelet function parameter ('omega0' for morlet, 'm' for paul
          and dog)
            
    :Returns:
       X : 2d numpy array
          transformed data
    """
    #import scipy.fft
    import numpy as np
    
    if x.ndim != 1:
        
        x_arr = np.asarray(x)- np.mean(x,axis=0)
        
        scales_arr = np.asarray(scales)
        
        if scales_arr.ndim != 1:
            raise ValueError('scales must be an 1d numpy array of list')
        
        w = angularfreq(N=x_arr.shape[0], dt=dt)
        
        if wf == 'dog':
            wft = dogft(s=scales_arr, w=w, order=p, dt=dt)
        elif wf == 'paul':
            wft = paulft(s=scales_arr, w=w, order=p, dt=dt)
        elif wf == 'morlet':
            wft = morletft(s=scales_arr, w=w, w0=p, dt=dt)
        else:
            raise ValueError('wavelet function is not available')
        
        WFT = np.tile(wft[:, :, np.newaxis, np.newaxis], (1,1,x_arr.shape[1],x_arr.shape[2]))

        X_ARR = np.empty((wft.shape[0], wft.shape[1],x_arr.shape[1],x_arr.shape[2]), dtype=complex128)

        x_arr_ft = np.fft.fft(x_arr,axis=0)

        for i in range(X_ARR.shape[0]):
            X_ARR[i] = np.fft.ifft(x_arr_ft * WFT[i],axis=0)
            
    else:
        
        x_arr = np.asarray(x) - np.mean(x)
        scales_arr = np.asarray(scales)

        #if x_arr.ndim != 1:
            #raise ValueError('x must be an 1d numpy array of list')

        if scales_arr.ndim != 1:
            raise ValueError('scales must be an 1d numpy array of list')

        w = angularfreq(N=x_arr.shape[0], dt=dt)

        if wf == 'dog':
            wft = dogft(s=scales_arr, w=w, order=p, dt=dt)
        elif wf == 'paul':
            wft = paulft(s=scales_arr, w=w, order=p, dt=dt)
        elif wf == 'morlet':
            wft = morletft(s=scales_arr, w=w, w0=p, dt=dt)
        else:
            raise ValueError('wavelet function is not available')

        X_ARR = mp.empty((wft.shape[0], wft.shape[1]), dtype=complex128)

        x_arr_ft = np.fft.fft(x_arr)

        for i in range(X_ARR.shape[0]):
            X_ARR[i] = np.fft.ifft(x_arr_ft * wft[i])
    
    return X_ARR

def icwt(X, dt, scales, wf='morlet', p=6):
    """Inverse Continuous Wavelet Tranform.
    The reconstruction factor is not applied.
    :Parameters:
       X : 2d array_like object
          transformed data
       dt : float
          time step
       scales : 1d array_like object
          scales
       wf : string ('morlet', 'paul', 'dog')
          wavelet function
       p : float
          wavelet function parameter
    :Returns:
       x : 1d numpy array
          data
    """  
    import numpy as np
    
    X_arr = np.asarray(X)
    
    scales_arr = np.asarray(scales)

    if X_arr.shape[0] != scales_arr.shape[0]:
        raise ValueError('X, scales: shape mismatch')

    # See (11), (13) at page 68
    X_ARR = np.empty_like(X_arr)
    
    for i in range(scales_arr.shape[0]):
        X_ARR[i] = X_arr[i] / np.sqrt(scales_arr[i])
    
    x = np.sum(np.real(X_ARR), axis=0)
   
    return x