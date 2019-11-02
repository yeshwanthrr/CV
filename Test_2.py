import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def load_data_mnist():
    images,labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    n_train= 60000 #The size of the training set
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    test_images = images[n_train:]
    test_labels = labels[n_train:]
    return (train_images.astype(np.float32)/255,train_labels.astype(np.float32),
            test_images.astype(np.float32)/255,test_labels.astype(np.float32))

train_imgs,train_lbls,test_imgs,test_lbls = load_data_mnist() 

labels_5 = np.where(train_lbls == 5)
immatrix = train_imgs[list(labels_5)]


def cust_pca(X=immatrix):
    
    X=immatrix
    num_data,dim = X.shape
    
    #calcualate mean
    mean_X = X.mean(axis=0)
    X = X-mean_X
    
    if dim>num_data:
        M = np.dot(X,X.T) #covariance matrix
        e,EV = np.linalg.eigh(M) #eigenvalues and eigne vectors
        tmp = np.dot(X.T,EV).T 
        V = tmp[::-1] #reverce to get the highest eigen vectors
        S = np.sqrt(e)[::-1] #reverse the vector to get highest eigen values
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        U,S,V = np.linalg.svd(X)
        V = V[:num_data]

    return V,S,mean_X #return projection matrix,variance and the mean

V,S,imgmean = cust_pca()

# show some images (mean and first 2 modes)
plt.figure()
plt.gray()
plt.subplot(2,4,1)
plt.imshow(imgmean.reshape(28,28))
for i in range(2):
    plt.subplot(2,4,i+2)
    plt.imshow(V[i].reshape(28,28))
    plt.show()
    
#--------------------2b---------------------------------------------

pca10 = V[:10]  
pca50 = V[:50]

test_img = immatrix[0]
#aa = ((test_img - imgmean)@pca10[0])
sum=0.0
for i in range(10):
    aa = ((test_img - imgmean)@pca10[i])*S[i]
    sum = sum + aa
reconstruct = imgmean+ abs(sum)

plt.imshow(test_img.reshape(28,28))
plt.imshow(reconstruct.reshape(28,28))

'''
aa=np.zeros((1,784),dtype=int)
bb=np.zeros((784,784),dtype=int)
for i in range(784):
    aa[i] = np.sum(np.multiply((test_img - imgmean),pca10[i]))
    bb[i] = np.dot(aa[i],pca10[i])
    sum = sum + bb[i]
'''

data_reduced = np.dot(test_img, pca10.T) # transform
data_original = np.dot(data_reduced, pca10) # inverse_transform
plt.imshow(reconstruct.reshape(28,28))

#test = test_img - data_original
