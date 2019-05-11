from __future__ import print_function,division
import os
import numpy as np
import scipy.io as sio
from sklearn.metrics.pairwise import euclidean_distances as ecdist
from sklearn.neighbors import NearestNeighbors
# from annoy import AnnoyIndex
from scipy import sparse
import timeit
from pyflann import *
from PIL import Image
from sklearn.utils import linear_assignment_
from sklearn.metrics import accuracy_score
import math

def get_accuracy(L1, L2):
    if L1.__len__() != L2.__len__():
        print('size(L1) must == size(L2)')

    Label1 = np.unique(L1)
    nClass1 = Label1.__len__()
    Label2 = np.unique(L2)
    nClass2 = Label2.__len__()

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()

    c = linear_assignment_.linear_assignment(-G.T)[:, 1]
    newL2 = np.zeros(L2.__len__())
    for i in range(nClass2):
        for j in np.nonzero(L2 == Label2[i])[0]:
            if len(Label1) > c[i]:
                newL2[j] = Label1[c[i]]

    return accuracy_score(L1, newL2),newL2

def create_affinity(X, knn, scale = None, alg = "annoy", savepath = None, W_path = None):
    N,D = X.shape
    if W_path is not None:
        if W_path.endswith('.mat'):
            W = sio.loadmat(W_path)['W']
        elif W_path.endswith('.npz'):
            W = sparse.load_npz(W_path)
    else:
        
        print('Compute Affinity ')
        start_time = timeit.default_timer()
        if alg == "flann":
            print('with Flann')
            flann = FLANN()
            knnind,dist = flann.nn(X,X,knn, algorithm = "kdtree",target_precision = 0.9,cores = 5);
            # knnind = knnind[:,1:]
        # elif alg == "annoy":
        #     print('with annoy')
        #     ann = AnnoyIndex(D, metric='euclidean')
        #     for i, x_ in enumerate(X):
        #         ann.add_item(i, x_)
        #     ann.build(50)
        #     knnind = np.empty((N, knn))
        #     dist = np.empty((N, knn))
        #     for i in range(len(X)):
        #         nn_i = ann.get_nns_by_item(i, knn, include_distances=True)
        #         knnind[i,:] = np.array(nn_i[0])
        #         dist[i,:] = np.array(nn_i[1])
        else:
            nbrs = NearestNeighbors(n_neighbors=knn).fit(X)
            dist, knnind = nbrs.kneighbors(X)

        row = np.repeat(range(N),knn-1)
        col = knnind[:,1:].flatten()
        if scale is None:
            data = np.ones(X.shape[0]*(knn-1))
        else:
            data = np.exp((-dist[:,1:]**2)/(2 * scale ** 2)).flatten() 

        W = sparse.csc_matrix((data, (row, col)), shape=(N,N),dtype=np.float)
        # W = (W + W.transpose(copy=True)) /2
        elapsed = timeit.default_timer() - start_time
        print(elapsed)         

        if isinstance(savepath,str):
            if savepath.endswith('.npz'):
                sparse.save_npz(savepath,W)
            elif savepath.endswith('.mat'):
                sio.savemat(savepath,{'W':W})
            
    return W
        # W =np.empty((N,D))
    #    KDTREE
    #    tree = KDTree(X, leaf_size=40)
    #    knnid = tree.query(X, k=knn,return_distance=False)
    #    row = np.repeat(range(N),knn)
    #    col = knnind.flatten()
    #    data = np.ones(N*knn)
    #    W = sparse.csc_matrix((data, (row, col)), shape=(N,N),dtype=np.float)
    #    W.setdiag(0)

def mode_nn(mode_index,X,K,C,l,knn,X_org,path,imsize):
    if not os.path.exists(path):
        os.makedirs(path)

    for k in range(K):
        tmp=np.asarray(np.where(l== k))
        if tmp.size !=1:
            tmp = tmp.squeeze()
        else:
            tmp = tmp[0]
        tmp_X = X[tmp,:]
        tmp_X_org = X_org[tmp,:]
        nbrs = NearestNeighbors(n_neighbors=knn, algorithm='brute').fit(tmp_X)
        knnd, knnidx = nbrs.kneighbors(C[[k],:])
        knnidx = knnidx.squeeze()
        mode_image = X_org[mode_index[k],:].astype(np.uint8).reshape(imsize)
        im = Image.fromarray(mode_image)
        savepath = path+'mode_'+str(k)
        im.save(savepath+'.png')
        knnidx = knnidx[1:]
        for i in range(knnidx.size):
            # print(i)
            # cluster_knn(k,im) = cellstr(imagelabels(im).annotation.filename);
            im = tmp_X_org[knnidx[i],:].astype(np.uint8).reshape(imsize)
            cluster_knn_im = Image.fromarray(im)
            savepath = path+'mode_'+str(k)+'_knn_'+str(i+1)
            cluster_knn_im.save(savepath+'.png')


def estimate_sigma(X,W,knn,N): 
    if N>70000:
        batch_size = 4560
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        sigma_square = 0
        for batch_A in range(num_batch):
            start1 = batch_A*batch_size
            end1 = min((batch_A+1)*batch_size, N)
            for batch_B in range(num_batch):
                start2 = batch_B*batch_size
                end2 = min((batch_B+1)*batch_size, N)
                print("start1 = %d|start2 = %d"%(start1,start2))
                pairwise_dists = ecdist(X[start1:end1],X[start2:end2],squared =True)
                W_temp = W[start1:end1,:][:,start2:end2]
                sigma_square = sigma_square+(W_temp.multiply(pairwise_dists)).sum()
                print (sigma_square)
        sigma_square = sigma_square/(knn*N)
        sigma = np.sqrt(sigma_square)
    else:  
        pairwise_dists = ecdist(X,squared =True)
        sigma_square = W.multiply(pairwise_dists).sum()
        sigma_square = sigma_square/(knn*N)
        sigma = np.sqrt(sigma_square)
    return sigma

def estimate_median_sigma(X,knn,batch_size=1028):
    n = len(X)
    batch_size = min(n,batch_size)
    # sample a random batch of size batch_size
    sample = X[np.random.randint(n, size=batch_size), :]
    # flatten it
    sample = sample.reshape((batch_size, np.prod(sample.shape[1:])))

    # compute distances of the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=knn).fit(sample)
    distances, _ = nbrs.kneighbors(sample)

    # return the median distance
    return np.median(distances[:,knn-1])
    
def validation_set(X,gnd_labels,K,val_frac):
    X_val = []
    gnd_val = []
    indices = []
    cluster_size_list = []
    for k in range(K):
        tmp=np.asarray(np.where(gnd_labels== k)).squeeze()
        cluster_size = tmp.size
        cluster_size_list.append(cluster_size)
        batch = cluster_size *val_frac
        batch = min(batch,cluster_size)
        indices_k = np.random.choice(tmp,int(batch))
        indices.append(indices_k)
        X_val.append(X[indices_k,:])
        gnd_val.append(gnd_labels[indices_k])
    imbalance = max(cluster_size_list)/min(cluster_size_list)
    print('dataset imbalance = %0.4f'%imbalance)
    X_val = np.vstack(X_val)
    gnd_val = np.hstack(gnd_val)
    indices = np.hstack(indices)
    return X_val,gnd_val,indices,imbalance
