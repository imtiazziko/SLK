#!/usr/bin/python 
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:09:40 2017

@author: ziko
"""
from __future__ import print_function,division
import numpy as np
import sys
import math
#import scipy.io as sio
from scipy.spatial import distance as dist
from sklearn.metrics.pairwise import euclidean_distances as ecdist
from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _init_centroids
#import multiprocessing
from bound_update import bound_update
import timeit
#import random


def normalizefea(X):
    """
    Normalize each row
    """
    feanorm = np.maximum(1e-14,np.sum(X**2,axis=1))
    X_out = X/(feanorm[:,None]**0.5)
    return X_out

def MS(X,s,tmp,c0,tol,maxit):
    """
    Mean-shift iteration until convergence
    """
    # print 'inside meanshift iterations.'
    for i in range(maxit):
        Y = dist.cdist(c0, X[tmp,:], 'euclidean')
        W = np.exp((-Y**2)/(2 * s ** 2))
        c1 = np.dot(W,X[tmp,:])/np.sum(W)
        if np.amax(np.absolute(c1-c0))<tol*np.amax(np.absolute(c0)):
            break
        else:
            c0 = c1.copy()
    return c1

def KM(X,tmp,s):
    """
    Mode using definition m_l = \max(x_p in X) \sum_q k(x_p,x_q)
    
    """
    tmp_size = tmp.size
    
    size_limit = 25000 # Decrease in case of memory error
    if tmp_size>size_limit: 
        batch_size = 1024
        Deg = []
        num_batch = int(math.ceil(1.0*tmp_size/batch_size))
        for batch_idx in range(num_batch):
            start = batch_idx*batch_size
            end = min((batch_idx+1)*batch_size, tmp_size)
            pairwise_dists = ecdist(X[tmp[start:end]],squared =True)
            W = np.exp(-pairwise_dists/(2* (s ** 2)))
            np.fill_diagonal(W,0)
            Deg_batch = np.sum(W,axis=1).tolist()
            Deg.append(Deg_batch)
        m = max(Deg)
        ind = Deg.index(m)
        mode_index = tmp[ind]
        c1 = X[[tmp[ind]],:]
    else:
        pairwise_dists = ecdist(X[tmp,:],squared =True)
        W = np.exp(-pairwise_dists/(2* (s ** 2)))
        np.fill_diagonal(W,0)
        Deg = np.sum(W,axis=1)
        ind = np.argmax(Deg)
        mode_index = tmp[ind]
        c1 = X[[tmp[ind]],:]

    return c1,mode_index


def km_le(X,M,assign,sigma):
    
    """
    Discretize the assignments based on center
    
    """
    
#    e_dist = dist.cdist(X,M,'euclidean')   
    e_dist = ecdist(X,M)          
    
    
    if assign == 'gp':
        g_dist =  np.exp(-e_dist**2/(2*sigma**2))
        l = g_dist.argmax(axis=1)
        energy = compute_km_energy(l,g_dist.T)
        print ('Energy of Kmode = ' + repr(energy))
    else:
        l = e_dist.argmin(axis=1)
        
    return l


def compute_km_energy(l,dist_c):
    """
     K-modes energy estimation
     
    """
    
#    dist_c_sum = np.sum(dist_c,axis=1)
    E = 0.0;
    for k in range(len(dist_c)):
        tmp = np.asarray(np.where(l== k)).squeeze()
        E -= np.sum(dist_c[k,tmp])
    return E

def compute_energy_lapkmode(X,C,l,W,sigma,bound_lambda):
    """
    compute Laplacian K-modes energy in discrete form
    
    """
    e_dist = ecdist(X,C,squared =True)
    g_dist =  np.exp(-e_dist/(2*sigma**2))
    pairwise = 0
    Index_list = np.arange(X.shape[0])
    for k in range(C.shape[0]):
        tmp=np.asarray(np.where(l== k))
        if tmp.size !=1:
            tmp = tmp.squeeze()
        else:
            tmp = tmp[0]
        # print('length of tmp ', len(tmp))
        # pairwise = pairwise - W[tmp,:][:,tmp].sum() # With potts values -1/0
        nonmembers = np.in1d(Index_list,tmp,invert =True) # With potts values 0/1
        pairwise = pairwise + W[tmp,:][:,nonmembers].sum()
    E_kmode = compute_km_energy(l,g_dist.T)
    print(E_kmode)
    E = (bound_lambda)*pairwise + E_kmode
    return E
    
def km_init(X,K,C_init):
    
    """
    Initial seeds
    """
    
    N,D = X.shape
    if isinstance(C_init,str):

        if C_init == 'kmeans_plus':
            M =_init_centroids(X,K,init='k-means++')
            l = km_le(X,M,None,None)
        elif C_init =='rndmeans':
            m = X.min(0); mm =X.max(0)
            a = (mm-m)*np.random.random((K,D))
            M = a+m[None,:]
            l = km_le(X,M,None,None)
        elif C_init =='rndsubset':
            M = X[np.random.choice(list(range(N)),K),:]
#            tmp = np.random.permutation(N)
#            M = X[tmp[0:K],:]
            l = km_le(X,M,None,None)
        elif C_init =='kmeans':
            kmeans = KMeans(n_clusters=K).fit(X)
            l =kmeans.labels_
            M = kmeans.cluster_centers_
    else:
        M=C_init;
        l = km_le(X,M,None,None)
    del C_init
    return M,l

def SLK_iterative(X,sigma,K,W,bound_ = False, method = 'KM', C_init = "kmeans_plus",**opt):
    
    """ 
    Proposed SLK method with iterative mode updates
    
    """
    
    start_time = timeit.default_timer()
    C,l =  km_init(X,K,C_init)
    assert len(np.unique(l)) ==K
#    D =C.shape[1]
    trivial_status = False
    z = []
    bound_E = []
    mode_index = [];
    tol = 1e-3
    for i in range(100):
        oldC = C.copy()
        oldl = l.copy()
        oldmode_index = mode_index
        if method == 'MS':
            print ('Inside meanshift update . ... ........................')
            for k in range(C.shape[0]):
                tmp=np.asarray(np.where(l== k))
                if tmp.size !=1:
                    tmp = tmp.squeeze()
                else:
                    tmp = tmp[0]
                C[[k],] = MS(X,sigma,tmp,C[[k],],1e-5,int(1e3))
            # print C.shape
        elif method== 'KM':
            mode_tmp = []
            for k in range(C.shape[0]):
                tmp=np.asarray(np.where(l== k))
                if tmp.size !=1:
                    tmp = tmp.squeeze()
                else:
                    tmp = tmp[0]
                C[[k],],m = KM(X,tmp,sigma)
                mode_tmp.append(m)
            mode_index = mode_tmp[:]

        elif method == 'SLK-BO' and i==0:
            print('Inside SLK-BO')
            mode_tmp = []
            for k in range(C.shape[0]):
                tmp=np.asarray(np.where(l== k))
                if tmp.size !=1:
                    tmp = tmp.squeeze()
                else:
                    tmp = tmp[0]
                C[[k],],m = KM(X,tmp,sigma)
                mode_tmp.append(m)
            mode_index = mode_tmp[:]
            
        elif method not in ['SLK-MS','SLK-BO','MS']:
            print(' Error: Give appropriate method from SLK-MS/SLK-BO')
            sys.exit(1)
        

        if bound_==True:
            bound_lambda = opt['bound_lambda']
            bound_iterations = opt['bound_iterations']
            manual_parallel = False # False use auto numpy parallelization on BLAS/LAPACK/MKL
            sqdist = ecdist(X,C,squared=True)
            unary = np.exp((-sqdist)/(2 * sigma ** 2))
            if method == 'SLK-BO':
                l,C,mode_index,z,bound_E = bound_update(-unary,X,W,bound_lambda,bound_iterations,manual_parallel)
            else:
                l,_,_,z,bound_E = bound_update(-unary,X,W,bound_lambda,bound_iterations,manual_parallel)
                
            if (len(np.unique(l))!=K):
                print('not having some labels')
                trivial_status = True
                l =oldl.copy();
                C =oldC.copy();
                mode_index = oldmode_index;
                break;
            
        else:    
            l = km_le(X,C,str('gp'),sigma)
        
        if np.linalg.norm(C-oldC,'fro') < tol*np.linalg.norm(oldC,'fro'):
          print('......Job  done......')
          break
          
    elapsed = timeit.default_timer() - start_time
    print(elapsed) 
    return C,l,elapsed,mode_index,z,bound_E,trivial_status

#if __name__ == '__main__':
#    main()

    
    
