# -*- coding: utf-8 -*-
"""
Created on Wed Oct 9 08:24:54 2019

@author: Yeshwanth
"""

#importing the required libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
#from scipy import ndimage
from matplotlib import rcParams


#Function to display image
def plot_image(img,title='Image',clr='gray'):
    plt.figure()
    plt.title(title)
    plt.imshow(img,cmap=clr)


#Function to load image
def load_image(file,title='Image'):
    img_bgr=cv2.imread(file,1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plot_image(img_rgb,title)#'RGB Image'
    return img_rgb

#Convert image to LAB
def convert2LAB(img):
    img_lab =  cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #split the channel components
    l_component,a_component,b_component = cv2.split(img_lab)
    
    #plot each component image
    plot_image(l_component,'L component')
    plot_image(a_component,'A component')
    plot_image(b_component,'B component')
    
    return l_component,a_component,b_component

#Gaussian filter and derivative components
def derivative_components(component):
    img_gauss = cv2.GaussianBlur(component,(5,5),5,5)
    
    #Derivative along dx
    img_dx = np.diff(img_gauss,axis=0)
    plot_image(img_dx,'Gaussian horizontal derivative')
    
    #Derivative along dy
    img_dy = np.diff(img_gauss,axis=1)
    plot_image(img_dy,'Gaussian Vertical derivative')
    
#Plot a 2d histogram
def compHistogram(compA,compB,img):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    hist, xbins, ybins = np.histogram2d(compA.ravel(),compB.ravel(),bins=100)
    plt.figure()
    #plt.imshow(hist,cmap='gray',interpolation = 'nearest',vmin=0,vmax=255)
    
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xbins[:-1] + 0.25, ybins[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')    
    
#Back projection
'''
def back_projection(img_src,img_tgt):
    #img_src=cv2.imread(img_src,1)
    #img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2LAB)
    #img_tgt = load_image(img_tgt,'Target Image')
    img_srca = cv2.cvtColor(img_src, cv2.COLOR_RGB2LAB)
    img_tgt_rgb= load_image(img_tgt)
    img_tgt = cv2.cvtColor(img_tgt_rgb, cv2.COLOR_RGB2LAB)
    tgt_hist = cv2.calcHist([img_tgt], [1, 2], None, [256, 256], [0, 256, 0, 256])
    tgt_bp = cv2.calcBackProject([img_srca],[1,2],tgt_hist,[0,256,0,256],1)
    
    #Removing Noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    tgt_bp = cv2.filter2D(tgt_bp,-1,kernel)
    
    #mark the image using binary threshold
    _, tgt_bp = cv2.threshold(tgt_bp, 255, 255, cv2.THRESH_BINARY_INV)
    tgt_bp = cv2.merge((tgt_bp, tgt_bp, tgt_bp))
    result = cv2.bitwise_and(img_src,tgt_bp)
    #plot_image(result,'Back Projection')
    result1 = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    plt.imshow(result1,cmap='gray')
'''

def back_projection(img_src,img_tgt):
    img_srca = cv2.imread(img_src)
    img_src = cv2.cvtColor(img_srca, cv2.COLOR_BGR2RGB)
    img_tgta = cv2.imread(img_tgt)
    img_tgt = cv2.cvtColor(img_tgta, cv2.COLOR_BGR2LAB)
    img_src1 = cv2.cvtColor(img_srca, cv2.COLOR_BGR2LAB)
    tgt_hist = cv2.calcHist([img_tgt], [1, 2], None, [256, 256], [0, 256, 0, 256])
    tgt_bp = cv2.calcBackProject([img_src1],[1,2],tgt_hist,[0,256,0,256],1)

    #Removing Noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    tgt_bp = cv2.filter2D(tgt_bp,-1,kernel)

    #mark the image using binary threshold
    _, tgt_bp = cv2.threshold(tgt_bp, 127, 255, cv2.THRESH_BINARY)
    tgt_bp = cv2.merge((tgt_bp, tgt_bp, tgt_bp))
    result = cv2.bitwise_and(img_src,tgt_bp)
    #plot_image(result,'Back Projection')
    result1 = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    plt.imshow(result1,cmap='gray')
    
#Histogram Equalization

def histEqualisation(l_component):
    eq_channels = []
    eq_channels.append(cv2.equalizeHist(l_component))

    eq_image = cv2.merge(eq_channels)
    #eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB)
    #plt.imshow(eq_image)
    #plt.show()

    rcParams['figure.figsize'] = 11 ,8
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(l_component,cmap="gray")
    ax[0].title.set_text('Original L')
    ax[1].imshow(eq_image,cmap="gray")
    ax[1].title.set_text('Equalized L')

def main():
    #1a-Read an RGB Image
    img_rgb = load_image("D:\\OneDrive\\OneDrive - TCDUD.onmicrosoft.com\\TCD Assignments\\Computer Vision\\bird flower.jpg")
    
    #1b- To convert RGB to LAB
    l_,a_,b_=convert2LAB(img_rgb)
    
    #1c -Derivative of x and Y component
    derivative_components(l_)
    
    
    #1d - Calculate the 2D_Histogram of a and b component
    compHistogram(a_,b_,img_rgb)
    
    #1e - BackProjection
    img_src_path = "D:\\OneDrive\\OneDrive - TCDUD.onmicrosoft.com\\TCD Assignments\\Computer Vision\\bird flower.jpg"
    img_tgt_path = "D:\\OneDrive\\OneDrive - TCDUD.onmicrosoft.com\\TCD Assignments\\Computer Vision\\flower.png"
    back_projection(img_src_path,img_tgt_path)
    
    #1f - Histogram Equalisation
    histEqualisation(l_)


if __name__ == '__main__':
    main()
    




    