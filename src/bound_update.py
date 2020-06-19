# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:38:23 2017

@author: ziko
"""
import numpy as np
import multiprocessing
import itertools
import scipy.sparse as sps
import timeit
from src.progressBar import printProgressBar
import math

SHARED_VARS = {}
SHARED_array = {}

def normalize(Q_in):
    maxcol = np.max(Q_in, axis=1)
    Q_in = Q_in - maxcol[:,np.newaxis]
    N = Q_in.shape[0]
    size_limit = 150000
    if N>size_limit:
        batch_size = 1280
        Q_out = []
        Q_out_2 = []
        num_batch = int(math.ceil(1.0*N/batch_size))
        for batch_idx in range(num_batch):
            start = batch_idx*batch_size
            end = min((batch_idx+1)*batch_size, N)
            tmp = np.exp(Q_in[start:end,:])
            tmp = tmp/(np.sum(tmp,axis=1)[:,None])
            Q_out.append(tmp)
        del Q_in
        Q_out = np.vstack(Q_out)
    else:
        Q_out = np.exp(Q_in)
        Q_out = Q_out/(np.sum(Q_out,axis=1)[:,None])
    
    return Q_out

def n2m(a):
    """
    Return a multiprocessing.Array COPY of a numpy.array, together
    with shape, typecode and matrix flag.
    """
    if not isinstance(a, np.ndarray): a = np.array(a)
    return multiprocessing.Array(a.dtype.char, a.flat, lock=False), tuple(a.shape), a.dtype.char, isinstance(a, np.matrix)

def m2n(buf, shape, typecode, ismatrix=False):
    """
    Return a numpy.array VIEW of a multiprocessing.Array given a
    handle to the array, the shape, the data typecode, and a boolean
    flag indicating whether the result should be cast as a matrix.
    """
    a = np.frombuffer(buf, dtype=typecode).reshape(shape)
    if ismatrix: a = np.asmatrix(a)
    return a

def mpassing(slices):
    
    i,k = slices
    Q_s,kernel_s_data,kernel_s_indices,kernel_s_indptr,kernel_s_shape = get_shared_arrays('Q_s','kernel_s_data','kernel_s_indices','kernel_s_indptr','kernel_s_shape')
    # kernel_s = sps.csc_matrix((SHARED_array['kernel_s_data'],SHARED_array['kernel_s_indices'],SHARED_array['kernel_s_indptr']), shape=SHARED_array['kernel_s_shape'], copy=False)
    kernel_s = sps.csc_matrix((kernel_s_data,kernel_s_indices,kernel_s_indptr), shape=kernel_s_shape, copy=False)
    Q_s[i,k] = kernel_s[i].dot(Q_s[:,k])
#    return Q_s

    
def new_shared_array(shape, typecode='d', ismatrix=False):
    """
    Allocate a new shared array and return all the details required
    to reinterpret it as a numpy array or matrix (same order of
    output arguments as n2m)
    """
    typecode = np.dtype(typecode).char
    return multiprocessing.Array(typecode, int(np.prod(shape)), lock=False), tuple(shape), typecode, ismatrix

def get_shared_arrays(*names):
    return [m2n(*SHARED_VARS[name]) for name in names]

def init(*pargs, **kwargs):
    SHARED_VARS.update(pargs, **kwargs)

def entropy_energy(Q,unary,kernel,bound_lambda,batch = False):
    tot_size = Q.shape[0]
    pairwise = kernel.dot(Q)
    if batch == False:
        temp = (unary*Q) + (-bound_lambda*pairwise*Q)
        E = (Q*np.log(np.maximum(Q,1e-20))+temp).sum()
    else:
        batch_size = 1024
        num_batch = int(math.ceil(1.0*tot_size/batch_size))
        E = 0
        for batch_idx in range(num_batch):
            start = batch_idx*batch_size
            end = min((batch_idx+1)*batch_size, tot_size)
            temp = (unary[start:end]*Q[start:end]) + (-bound_lambda*pairwise[start:end]*Q[start:end])
            E = E+(Q[start:end]*np.log(np.maximum(Q[start:end],1e-20))+temp).sum()
                
    return E

def get_S_discrete(l,N,K):
    x = range(N)
    temp =  np.zeros((N,K),dtype=float)
    temp[(x,l)]=1
    return temp
            
def bound_update(unary,X,kernel,bound_lambda,bound_iteration =20, batch = False, manual_parallel =False):
    
    """

    """
    start_time = timeit.default_timer()
    print("Inside Bound Update . . .")
    N,K = unary.shape;
    oldE = float('inf')

    # Initialize the unary and Normalize
    if manual_parallel == False:
        # print 'Parallel is FALSE'
        Q = normalize(-unary) 
        for i in range(bound_iteration):
            printProgressBar(i + 1, bound_iteration,length=12)
            additive = -unary
            mul_kernel = kernel.dot(Q)
            Q = -bound_lambda*mul_kernel
            additive = additive -Q
            Q = normalize(additive)
            E = entropy_energy(Q,unary,kernel,bound_lambda,batch)
            print('entropy_energy is ' +repr(E) + ' at iteration ',i)
            report_E = E
            if (i>1 and (abs(E-oldE)<= 1e-6*abs(oldE))):
                print('Converged')
                break

            else:
                oldE = E.copy(); oldQ = Q.copy(); report_E = E      
    else:
        print('Manual Parallel is TRUE')
        Q = normalize(-unary)
        
        init(kernel_s_data = n2m(kernel.data))
        init(kernel_s_indices = n2m(kernel.indices))
        init(kernel_s_indptr = n2m(kernel.indptr))
        init(kernel_s_shape = n2m(kernel.shape))

        irange = range(N);
        krange = range(K);
        
        for i in range(bound_iteration):
            printProgressBar(i + 1, bound_iteration,length=15)
            additive = -unary
            init(Q_s = n2m(Q))
            pool = multiprocessing.Pool(processes=5,initializer=init, initargs = list(SHARED_VARS.items()))
            pool.map(mpassing,itertools.product(irange,krange))
            _,Q = get_shared_arrays('kernel_s_indptr','Q_s')
            Q = -bound_lambda*Q
#            assert (Q.all()==SHARED_array['Q_s'].all())
            additive -= Q
            Q = normalize(additive)
            pool.close()
            pool.join()
            pool.terminate()
            E = entropy_energy(Q,unary,kernel,bound_lambda,batch)
#            print ('entropy_energy is ' +repr(E) + ' at iteration ',i)
            if (i>1 and (abs(E-oldE)<=1e-4*abs(oldE))):
                print('Converged')
                break
            else:
                oldE = E.copy(); oldQ = Q.copy(); report_E = E  
                        
    elapsed = timeit.default_timer() - start_time
    print('\n Elapsed Time in bound_update', elapsed)
    l = np.argmax(Q,axis=1)
    ind = np.argmax(Q,axis=0)
    C= X[ind,:]
    return l,C,ind,Q,report_E

