import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np
import math
from PIL import Image, ImageEnhance
import pylab #import *
#import numpy as np
from scipy.ndimage import filters
#from skimage import io
import glob, os
def line(rows,cols,deg):
    out = np.zeros([rows,cols])
    center = [int((rows+1)/2)-1,int((cols+1)/2)-1]
    r = [0,0]
    if cols * abs(math.tan(deg))>=rows:
        add = (1/math.tan(deg),1)
    else:
        add = (1,math.tan(deg))
    while center[0]+int(r[0]+0.5) < rows and center[1]+int(r[1]+0.5) < cols:
        out[center[0]+int(r[0]+0.5),center[1]+int(r[1]+0.5)] = 1/35
        out[center[0]-int(r[0]+0.5),center[1]-int(r[1]+0.5)] = 1/35
        r[0] = r[0]+add[0]
        r[1] = r[1]+add[1]
    return out

def linedrawing(img,n,x,y,size=0):
    ## img: ori-img
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if size != 0:
        img = cv2.GaussianBlur(img,(size,size),0)
    rows = len(img)
    cols = len(img[0])

    G = gradient(img)

    Li = [line(x,y,math.pi/n*i) for i in range(n)]
               
    Gi = [cv2.filter2D(G,-1,Li[i]) for i in range(n)]

    for i in range(rows):
        for j in range(cols):
            response = [Gi[k][i,j] for k in range(n)]
            index = response.index(max(response))
            for k in range(n):
                if k != index:
                    Gi[k][i,j] = 0

    result = sum([cv2.filter2D(Gi[i],-1,Li[i]) for i in range(n)])
    result = result / max(result.reshape((rows*cols,1)))[0]
    result = np.ones((rows,cols))-result
    return result

def gradient(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absx = cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absx,0.5,absy,0.5,0)

def canny(img,low):
    #img = plt.imread(name,0)
    img = cv2.GaussianBlur(img,(3,3),0)
    rows = len(img)
    cols = len(img[0])
    return np.ones((rows,cols))*255-cv2.Canny(img,low,low*3)

def XDOG(img,sigma,k,tau,e,phi):
    D = cv2.GaussianBlur(img,(int(sigma**0.5),int(sigma**0.5)),sigma)-tau*cv2.GaussianBlur(img,(int((k*sigma)**0.5),int((k*sigma)**0.5)),k*sigma)
    '''
    for i in range(len(D)):
        for j in range(len(D[0])):
            if D[i][j]>=e:
                D[i][j] = 1
            else:
                D[i][j] = 1+math.tanh(phi*(D[i][j]-e))
    '''
    return D

def XDOG2(files1):
    Gamma = 0.97
    Phi = 200
    Epsilon = 0.1
    k = 2.5
    Sigma = 1.5

    im = Image.open(files1).convert('L')
    im = pylab.array(ImageEnhance.Sharpness(im).enhance(3.0))
    im2 = filters.gaussian_filter(im, Sigma)
    im3 = filters.gaussian_filter(im, Sigma* k)
    differencedIm2 = im2 - (Gamma * im3)
    (x, y) = pylab.shape(im2)
    for i in range(x):
        for j in range(y):
            if differencedIm2[i, j] < Epsilon:
                differencedIm2[i, j] = 1
            else:
                differencedIm2[i, j] = 250 + pylab.tanh(Phi * (differencedIm2[i, j]))


    gray_pic=differencedIm2.astype(np.uint8)
    #final_img = Image.fromarray( gray_pic)
    #cv2.imshow("--",gray_pic)
    #cv2.waitKey(0)
    return gray_pic

if __name__ == '__main__':      
    img = plt.imread('data/3--17.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    result = linedrawing('data/3--17.jpg',8,x=30,y=30,size=0)
    cv2.imshow('',result)
    cv2.waitKey(0)
    '''
    D = XDOG(img,30,2,1,100,0)
    cv2.imshow("--",D)
    cv2.waitKey(0)
    
    XOG2('data/3--17.jpg')
    '''
            
