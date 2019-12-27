
import cv2
from PIL import Image
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import cg



# Transfer the texture so that it matches the tone image by sovling linear equation
def renderTexture(H,J,lam=0.2):
    ''' J is the tonal pencil texture, H is the human drawn tonal pattern
        Both of them are numpy array
        lam is the weight, which is recommended to be set as 0.2 '''
    #H = resize(H,J)
    
    (h,w) = np.shape(J)
    H = cv2.resize(H, (w, h), interpolation=cv2.INTER_CUBIC)
    # prevent log(0)
    epsilon = 1e-5 
    # map [0,255] to(0,1]
    H = np.float64(H+epsilon)
    J = np.float64(J+epsilon)
    # transform to vector
    H_vec = np.reshape(H, w*h)
    J_vec = np.reshape(J, w*h)
    # take the logarithm
    logH = np.log(H_vec)
    logJ = np.log(J_vec)
    # the x&y directional deritive for vectorlized image
    idt = np.ones(w*h)
    Dx = spdiags(np.array([-idt,idt]), np.array([0,1]), w*h, w*h)
    Dy = spdiags(np.array([-idt,idt]), np.array([0,w]), w*h, w*h)
    # compute matrix A and b
    A = spdiags(logH*logH, 0, w*h, w*h) + lam*(Dx.transpose().dot(Dx)+Dy.transpose().dot(Dy))
    b = logH * logJ
    #A_0 = spdiags(logH, 0, w*h, w*h) + lam*(Dx.transpose().dot(Dx)+Dy.transpose().dot(Dy))
    #b_0 = logJ
    #A = np.multiarray(A_0.transpose(),A_0)
    #b = np.multiarray(A_0.transpose(),b)
    # conjugate gradient
    beta, _ = cg(A, b)
    Beta = beta.reshape(h, w)
    # the tone map
    T = H**Beta
    return T

def resize(H,J):
    '''adjust the array H to be the same as J'''
    H_im = Image.fromarray(H, 'L')
    J_im = Image.fromarray(J, 'L')
    H_im = H_im.resize(J_im.size)
    H = np.array(H_im)
    return H

def convertImage(im):
    '''convert pillow image (L channel) to numpy arrow'''
    im_array = np.array(im)
    im_Luv = cv2.cvtColor(im_array, cv2.COLOR_RGB2Luv)
    im_L = im_Luv[:,:,0]
    return im_L


    
    
    
