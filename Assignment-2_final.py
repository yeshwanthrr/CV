# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:46:45 2019

@author: Yeshwanth
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import seaborn as sns

def load_data_mnist():
    images,labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    n_train= 60000 #The size of the training set
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    test_images = images[n_train:]
    test_labels = labels[n_train:]
    return (train_images.astype(np.float32)/255,train_labels.astype(np.float32),
            test_images.astype(np.float32)/255,test_labels.astype(np.float32))


def selectTrainingSet(n=5):
    train_imgs,train_lbls,test_imgs,test_lbls = load_data_mnist()
    labels_n = np.where(train_lbls == n)
    imgmatrix = train_imgs[list(labels_n)]
    return train_imgs,train_lbls,test_imgs,test_lbls,imgmatrix

#Global DataSet
train_imgs,train_lbls,test_imgs,test_lbls,imgmatrix = selectTrainingSet()  

def cust_pca(X=imgmatrix):
    
    X=imgmatrix
    num_data,dim = X.shape
    
    #calcualate mean
    mean_X = X.mean(axis=0)
    X = X-mean_X
    #for higher dimension images
    if dim>num_data:
        covM = np.dot(X,X.T) #covariance matrix
        eg,EV = np.linalg.eigh(covM) #eigenvalues and eigne vectors
        tmp = np.dot(X.T,EV).T  #vector multiplication to match the dimension
        egV = tmp[::-1] #reverce to get the highest eigen vectors
        egS = np.sqrt(eg)[::-1] #reverse the vector to get highest eigen values
        for i in range(egV.shape[1]):
            egV[:,i] /= egS
    else:
        U,egS,egV = np.linalg.svd(X)
        egV = egV[:num_data]

    return egV,egS,mean_X #return projection matrix,variance and the mean

#egVector,eigen,imgmean = cust_pca()

def reconstImage(egVector,eigen,imgmean,test_img):
    #pca = egVector[:,n_comp]
    pca = egVector
    return np.dot(np.dot(test_img - imgmean,pca.T),pca) + imgmean

'''
pca_reconstruct_10 = reconstImage(egVector,eigen,10,imgmean,test_img)
pca_reconstruct_50 = reconstImage(egVector,eigen,50,imgmean,test_img)
pca_reconstruct_all = reconstImage(egVector,eigen,784,imgmean,test_img)

img_dict = {0:('Test Image',test_img),1:('PC10',pca_reconstruct_10),2:('PC50',pca_reconstruct_50)}
plotImages(img_dict)

'''

'''
test_img_scale =  []

for i in range(10):
    label_i = np.where(train_lbls == i)
    test_img_scale.append(train_imgs[label_i[0][0]])

comb_img = np.hstack( (np.asarray([ i.reshape(28,28) for i in test_img_scale ])))
comb_img_2 = np.vstack([comb_img]*4)
plt.imshow(comb_img_2)

a,b = comb_img_2.shape
'''
def ssd(target_img,imgmean):
    sd_vector = []
    for i in range(0,target_img.shape[0],28):
        for j in range(0,target_img.shape[1],28):
            #Calcuate ssd distance matrix
            sd_vector.append(np.sum(np.square(np.subtract(target_img[i:28+i,j:28+j],imgmean.reshape(28,28)))))
    #Plot Heat map of ssd
    plt.figure()
    sns.heatmap(np.array(sd_vector).reshape(4,10),cmap='gray')
    
    
        
#ssd(target_img,imgmean)

def dffs(target_img,pca_reconstruct,imgmean):
    df_vector = []
    for i in range(0,target_img.shape[0],28):
        for j in range(0,target_img.shape[1],28):
            #Calcuate dffs distance matrix
            df_vector.append(np.sqrt(np.sum(np.square(np.subtract(np.subtract(target_img[0+i:28+i,j:28+j],imgmean.reshape(28,28)),pca_reconstruct.reshape(28,28)))))) 
    #Plot Heat map of ssd
    plt.figure()
    sns.heatmap(np.array(df_vector).reshape(target_img.shape[0]/28,10),cmap='gray')
    
#dffs(target_img,pca_reconstruct_all,imgmean)


def plotImages(d):
    #plt.figure()
    #fig.add_subplot(2,4,1)
    #plt.imshow(test_img.reshape(28,28))
    #plt.title("Test Image")
    fig = plt.figure(figsize=(8,8))
    plt.gray()
    fig = plt.figure(figsize=(8,8))
    for i in d.keys():
        fig.add_subplot(2,4,i+1)
        plt.imshow(d[i][1].reshape(28,28))
        plt.title(d[i][0])
    plt.show()
'''
def generateTestImage():
    test_img_scale =  []

    for i in range(10):
        label_i = np.where(train_lbls == i)
        test_img_scale.append(train_imgs[label_i[0][0]])

    comb_img = np.hstack( (np.asarray([ i.reshape(28,28) for i in test_img_scale ])))
    comb_img_2 = np.vstack([comb_img]*4)
    plt.imshow(comb_img_2)
    
    return comb_img_2
'''
def generateTestImage():
    test_img_scale =  []

    for i in range(4):
        for j in range(10):
            label_i = np.where(train_lbls == j)
            test_img_scale.append(train_imgs[label_i[0][i]])
    
    temp = np.hstack( (np.asarray([ i.reshape(28,28) for i in test_img_scale ])))
    
    test_img_scale =  []
    for i in range(0,temp.shape[1],280):
        test_img_scale.append(np.array(temp[0:28,0+i:280+i]))
    
    test_image = np.vstack(x for x in test_img_scale)
    
    return test_image

def generateTestImgPca(n):
    tlab = np.where(test_lbls == 5)
    tst_img = test_imgs[list(tlab)]
    test_img = tst_img[10]
    return test_img



def main():
	#Generate Eigenvalues,eigenvectors and mean image
	egVector,eigen,imgmean = cust_pca()
	
	#Plot mean and first two eigenvectors
	img_dict = {0:('Mean Image',imgmean),1:('First eigenvector',egVector[0]),2:('Second eigenvector',egVector[1])}
	plotImages(img_dict)
	
	#Test image for reconstruction
	test_img = generateTestImgPca(5)
	
	#pca components of 10 and 50
	pca_reconstruct_10 = reconstImage(egVector[:10],eigen,imgmean,test_img)
	pca_reconstruct_50 = reconstImage(egVector[:50],eigen,imgmean,test_img)
	#pca_reconstruct_all = reconstImage(egVector,eigen,imgmatrix.shape[0],imgmean,test_img)
	
	#Plot pca and mean
	img_dict = {0:('Test Image',test_img),1:('PC10',pca_reconstruct_10),2:('PC50',pca_reconstruct_50)}
	plotImages(img_dict)
	
	#Test Image for ssd and dffs
	target_img = generateTestImage()
	
	#Heat map of dffs
	dffs(target_img,pca_reconstruct_50,imgmean)
	
	#Heat map of ssd
	ssd(target_img,imgmean)
    
if __name__ == '__main__':
    main()
    
#test_img = generateTestImage()

'''
img_dict = {0:('Test Image',test_img),1:('PC10',reconst_img_pc10),2:('PC50',reconst_img_pc50)}

#Show original Image and pca10 and pca50 reconstruction image
#plt.figure()
fig = plt.figure(figsize=(8,8))
plt.gray()
#fig.add_subplot(2,4,1)
#plt.imshow(test_img.reshape(28,28))
#plt.title("Test Image")
fig = plt.figure(figsize=(8,8))
for i in img_dict.keys():
    fig.add_subplot(2,4,i+1)
    plt.imshow(img_dict[i][1].reshape(28,28))
    plt.title(img_dict[i][0])
plt.show()
    
    

egV,egS,imgmean = cust_pca()


 



train_imgs,train_lbls,test_imgs,test_lbls = load_data_mnist() 

labels_5 = np.where(train_lbls == 5)
imgmatrix = train_imgs[list(labels_5)]
'''