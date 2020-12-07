# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:07:14 2020

@author: Gustavo Eduardo MEreles Menesse - FaCEN-UNA
"""

from biospeckle_tools_Gustavo import *

def graphmhi(data,THR = 10):
    import numpy as np
    
    m = data.shape[0]
    
    T = np.zeros(data.shape)
    
    MHI = np.zeros((data.shape[1],data.shape[2]))
    
    k = 255/sum(np.arange(0,m,1))
    
    for i in range(m-1):
        
        S = data[i]-data[i+1]

        T[i] = abs(S) > THR

        MHI += T[i]*k*(m-1-i)

    return MHI

def graphmhi_lag(data,lag=1,exp=1,THR = 10):
    import numpy as np
    
    m = data.shape[0]
    
    T = np.zeros(data.shape)
    
    MHI = np.zeros((data.shape[1],data.shape[2]))
    
    k = 255/sum(np.arange(0,m,1))
    
    for i in range(m-lag):
        
        S = data[i]-data[i+lag]

        T[i] = abs(S) > THR*(lag)**exp

        MHI += 255*T[i]*k

    return MHI

def graphim(data,THR = 10):
    import numpy as np
    
    m = data.shape[0]
    
    GIM = np.zeros((data.shape[1],data.shape[2]))

    S = np.zeros((data.shape[1],data.shape[2]))
    
    for i in range(m-1):
        
        S = (data[i]-data[i+1])**2 + S

    GIM = S/(m-1)

    return GIM

def Analize_all_graph(method='MHI',NN=100):
    
    import os 
    
    import numpy as np

    allfiles = os.listdir(os.getcwd())

    Dirs = []

    for i in allfiles:
        Dir = os.path.isdir(i)

        if Dir == True:
            if str('.') not in i:
                Dirs.append(i)

    print('Available folders')

    for i in range(0,len(Dirs)):
        print(i,'-',Dirs[i])

    n = int(input('Elija una carpeta ingresando su numero (int):'))

    allfiles=os.listdir(Dirs[n])

    files = []

    for i in allfiles:
            Dir = os.path.isdir(i)

            if Dir != True:
                 if (str('.csv') not in i) and str('.txt') not in i:
                    files.append(i)
    
    files=sorted(files)

    print(files)
    
    for i in range(len(files)):

        DIR = Dirs[n] + '//' + files[i]
        
        if i==0:
            
            data= datapack(DIR)

            DataMap = np.zeros([len(files),data.shape[1],data.shape[2]])

        else:
            
            data= datapack(DIR)

        data[data<4] = 0
        
        print('#Analizing ', DIR)
        
        if method=='MHI':
            
            Map = graphmhi(data[1:NN],10)
        
        if method=='MHI_LAG':
            Map = graphmhi_lag(data[1:NN],5,15)
            
        if method=='IM':
            
            Map = graphim(data[1:NN])
        
        DataMap[i] = Map
        
    return DataMap

def stscorr(data,t0):
    ''' Spatial-temporal speckle correlation
    
    [1] ZDUNEK, A. et al. New nondestructive method based on spatial-temporal
%      speckle correlation technique for evaluation of apples quality during
%      shelf-life. International Agrophysics, v. 21, n. 3, p. 305-310, 2007. '''
    
    import numpy as np
    m,k,l = data.shape
    
    C = np.zeros(m)
    
    LL = np.arange(0,m,1)-t0
    
    Ltau = LL*tau
    
    A = data[t0].flatten()
    
    A = (A-A.mean())/A.std()
    
    for i in range(m):

        B = data[i].flatten()
        
        B = (B-B.mean())/B.std()
        
        C[i]= (np.dot(B.T, A)/B.shape[0])
        
       # C[i]= np.corrcoef(A,B,rowvar=False)[0,1:]
            
        #C[i] = np.correlate(data[0,:,:].flatten(),data[i,:,:].flatten())
    
    return C

def COOM(THSP,plot=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.feature import greycomatrix
    
    matrix_coocurrence = greycomatrix(THSP, [1], [np.pi/2], levels=255+1, normed=False, symmetric=False);

    if plot!=False:
        plt.figure(figsize=(10,10));

        plt.imshow(matrix_coocurrence[:,:,0,0],cmap='jet'); plt.clim(0, 100);
        plt.colorbar()
    
        Guar=input('Desea guardar una imagen de la matriz? (y/n)')

        while Guar!='y' and Guar!='n':
            Guar=input('Responda con y o n. Desea guardar una imagen de la matriz? (y/n)')

        if Guar=='y':
            Nom=input('Introduzca el nombre con el que desea guardar la imagen')
            plt.savefig(Nom + '.png')

    return  matrix_coocurrence

def im1(COM):
    '''Calculates the Inertia Moment of the Co-ocurrence Matrix'''
    import numpy as np
    w,h,a,b = np.shape(COM)

    SCOM=0
    E=0

    for i in range(w):
        for j in range(h):
            SCOM += COM[i,j]

    for i in range(w):
        for j in range(h):
            C = COM[i,j]*((i-j)**2)
            E += C/SCOM

    return E[0][0]

def Sel_Method(Data,DIR,AM='none'):
        '''Obtein the parameters of a ROI, opocite vertices point of a rectangle and the center point (X,Y)'''
        import cv2,os
        import numpy as np
        
        if AM !='none':
            
            uint_img = np.array(Data).astype(np.uint16)

            im = uint_img
        
        else:
            
            if DIR!='':

                allfiles=os.listdir(os.getcwd()+'\\'+ DIR)

            else:
                allfiles=os.listdir(os.getcwd())

                print('We will work with the images in',os.getcwd())

            #Crea una lista con todos los archivos de allfiles con extension bmp
            oimlist=[filename for filename in allfiles if  filename[-4:] in [".bmp",".BMP"]]

            #Ordenar las imagenes de manera ascendente
            oimlist=sorted(oimlist,key=lambda f: int(''.join(filter(str.isdigit, f))))

            #####################################################
            #Abre y carga las imagenes
            im = cv2.imread(os.getcwd()+'\\'+ DIR + '\\' + oimlist[5])
        
        r = []
        
        #Toma la resolución de la primera imagen de la lista
        w,h = Data.shape
        
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
        
        #Cierran las ventanas donde se abrio la ROI selector
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    #Redefine la región seleccionada para las resolución original de las imágenes
        r=[f*r[0],f*r[1],f*r[2],f*r[3]]
        
        X,Y = int(np.round(r[0]+0.5*r[2])),int(np.round(r[1]+.5*r[3]))
        
        Ra = int(input('Introduce the radio of the analysis region (int): '))
        
        return X,Y,Ra

def stscorrTHSP(data,t0=0):
    ''' Spatial-temporal speckle correlation
    
    [1] ZDUNEK, A. et al. New nondestructive method based on spatial-temporal
%      speckle correlation technique for evaluation of apples quality during
%      shelf-life. International Agrophysics, v. 21, n. 3, p. 305-310, 2007. '''
    
    import numpy as np
    k,l = data.shape
    
    C = np.zeros(l)
 
    A = data[:,t0].flatten()
    
    A = (A-A.mean())/A.std()
    
    for i in range(l):

        B = data[:,i].flatten()
        
        B = (B-B.mean())/B.std()
        
        C[i]= (np.dot(B.T, A)/B.shape[0])
        
       # C[i]= np.corrcoef(A,B,rowvar=False)[0,1:]
            
        #C[i] = np.correlate(data[0,:,:].flatten(),data[i,:,:].flatten())
    
    return C

def thsp_gauss(DATA,DIR,r=[],P=[], N=[],AM='none',plot=False,plotGauss=True,savefig=False):
    '''Give the Time History Speckle from a DataPack (Time, Row, Column) 3-d array taking random points of the datapack using a gaussian distribution and the numer of samples wanted
    '''
    
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    #arr = DATA[0,:,:]-DATA[0,:,:].mean();
    
    if AM!='none':
        arr = AM
    
    else:
        arr = DATA[0,:,:]
        
    c,f=arr.shape
    
    if P==[]:
        
        X,Y,Ra = Sel_Method1(arr,DIR,AM=AM)
        
    else:
        
        X,Y,Ra= P[0],P[1],P[2]
    
    if N==[]:
        N = int(4*Ra)
    
    #Define the standart deviation of the distribution
    sigma1= Ra/(2*f)
    sigma2= Ra/(2*c)
    mu=0
    
    if r!=[]:
        FF=r[0]
        CC=r[1]
        F=[]
        C=[]
    
    else:
        FF=(f*np.random.normal(mu, sigma1, N)).round(0)
        CC=(c*np.random.normal(mu, sigma2, N)).round(0)
        F=[]
        C=[]
    
    #Make sure that the points randomly selected are conteined in the image
    for i in range(0,len(FF)):
        if (X+FF[i])<=0:
            FF[i]=abs(X+FF[i])
            F.append(FF[i])
            
        elif (X+FF[i])>=(f-1):
            FF[i]=f-abs(f-(FF[i]+X))
            F.append(FF[i])
        else:
            F.append(X+FF[i])
            
    for i in range(0,len(CC)):
        if (Y+CC[i])<=0:
            CC[i]=abs(Y+CC[i])
            C.append(CC[i])
            
        elif (Y+CC[i])>=(c-1):
            CC[i]=c-abs(c-(CC[i]+Y))
            C.append(CC[i])
        else:
            C.append(Y+CC[i])

    
    for p in range(0,N):
        if p==0:
                THSP=np.zeros((N,DATA.shape[0]))
                
        for i in range(0,DATA.shape[0]):
            THSP[p,i] = DATA[i,int(C[p]-1),int(F[p]-1)]

    THSP=np.transpose(THSP)
    TH= np.array(THSP, dtype=np.uint8)
    
    if savefig !=False:
        out = Image.fromarray(TH)
        out.save('THSP.bmp')  
    
    if plotGauss!=False:
        #print(len(F),len(C))
        #print(max(F),min(F),max(C),min(C))
        #Visualize the sample points
        fig, axs = plt.subplots(1, 2, figsize=(20,8));
        
        axs[0].imshow(arr,cmap='gray', vmin=0, vmax=255)
        axs[0].scatter(F, C, c='r', s=5) 
        axs[0].set_title('RAW SPECKLE')
        
        axs[1].imshow(THSP.T,cmap='gray', vmin=0, vmax=255);
        axs[1].set_title('THSP')
        
        plt.show();
    
    return TH,[X,Y,Ra],[FF,CC]

