import numpy as np
import cv2
from skimage import io, color, filters, transform, exposure
import matplotlib
import matplotlib.pyplot as plt
import math as ma
import gf
from skimage.color import rgb2gray


def ToneMap(ori_img,ua=105,ub=225 ,sigb = 9,mud  =90,sigd = 11):
    ## input: origin image
    ## ua = 105,ub = 225
    ### bright 
    ##sigb = 9

    ### dark
    ## mud = 90
    ## sigd = 11

    omega = [[29,29,42],[11,37,52],[2,22,76]]
    [w1,w2,w3] = omega[1]

    yuvimg = color.rgb2yuv(ori_img)
    img = yuvimg[:,:,0]
    histarget = np.zeros(256)
    total = 0

    for i in range(256):
        if(ua <=i <= ub):
            p = 1/(ub-ua)
        else:
            p = 0
        t1 = (1/sigb)*np.exp(-(255-i)/sigb)
        t2 = p
        t3 = (1/np.sqrt(2*ma.pi*sigd))*np.exp(-pow((i-mud),2)/(2*sigd*sigd))*0.01
        targ = w1*t1+w2*t2+w3*t3
        histarget[i] = targ
        total+=targ

    histarget = histarget /total

    # hist to target hist

    ## cdf of target hist
    cdftarget  = np.cumsum(histarget)

    oriHist = exposure.histogram(img, nbins=256)
    # CDF of original:
    cdfori = np.cumsum(oriHist/ np.sum(oriHist))

    histind = np.zeros(256)
    for v in range(256):
        dist = np.abs(cdftarget - cdfori[v])
        histind[v] =  np.argmin(dist)
    histind = histind / 256
    J = histind[(255 * img).astype(np.int)]
    # smooth:
    Smooth_J = filters.gaussian(J, sigma=np.sqrt(2))
    return Smooth_J

def GF(ori_img):
    lum = rgb2gray(ori_img)
    r = 8
    eps = 0.05
    girlsm = gg.guided_filter(lum, lum, r, eps)
    plt.axis('off')
    plt.imshow(girlsm, cmap='gray')
    #plt.savefig("f1.png")
    plt.show()
    return 

if __name__ == '__main__':
    ori_img = io.imread('data/2--32.jpg')
    J = ToneMap(ori_img)
    plt.imshow(J, cmap='gray')
    #plt.savefig("tone2.jpg")
    plt.show()
    GF(ori_img)