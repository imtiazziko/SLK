from __future__ import print_function,division
import argparse
import os,sys
import scipy.io as sio
import numpy as np
from src.SLK_iterative import SLK_iterative
from src.SLK import SLK, normalizefea, km_init
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from src.util import get_accuracy, Logger
import src.util as util
import timeit
import random
from src.dataset_load import read_dataset, dataset_names
def main(args):

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    dataset = args.dataset
    data_dir = './data/'

    # SLK options BO/MS/Means
    SLK_option = args.SLK_option

    #  Save?
    mode_images = args.mode_images #  save mode images in a directory?
    saveresult = args.saveresult  #  save results?

    log_path = os.path.join(data_dir,SLK_option+'_'+dataset+'_log_.txt')
    sys.stdout = Logger(log_path)

    #   Give data matrix in samples by feature format ( N X D)
    X, gnd_labels, K, sigma, X_org, knn = read_dataset(dataset,data_dir)

    # Normalize Features
    X =normalizefea(X)

    N,D =X.shape

    #####Validation Set for tuning lambda and initial K-means++ seed.
    #### However you can set value of lambda and initial seed empirically and skip validation set #######

    val_path = data_dir + dataset + '_val_set.npz'

    if not os.path.exists(val_path):
        X_val,gnd_val,val_ind,imbalance = util.validation_set(X,gnd_labels,K,0.1)
        np.savez(val_path, X_val = X_val, gnd_val = gnd_val, val_ind = val_ind)
    else:
        data_val = np.load(val_path)
        X_val = data_val['X_val']
        gnd_val = data_val['gnd_val']
        val_ind = data_val['val_ind']

    ##    # Build the knn kernel
    start_time = timeit.default_timer()

    aff_path = data_dir + 'W_'+str(knn)+'_'+ dataset+'.npz'
    alg = None
    if N>100000:
        alg = "flann"

    if not os.path.exists(aff_path):
        W = util.create_affinity(X, knn, scale = None, alg = alg, savepath = aff_path, W_path = None)
    else:
        W = util.create_affinity(X, knn, W_path = aff_path)

    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    ###### Run SLK#################################

    bound_ = args.bound # Setting False only runs K-modes
    bound_it = 5000

    if sigma is None:
        sigma = util.estimate_sigma(X,W,knn,N)
    #        sigma = util.estimate_median_sigma(X,knn) # Or this

    # Initial seed path from kmeans++ seed
    init_C_path = data_dir+dataset+'_C_init.npy'
    if not os.path.exists(init_C_path):
        C_init,_ = km_init(X,K,'kmeans_plus')
        np.save(init_C_path,C_init)
    else:
        C_init = np.load(init_C_path) # Load initial seeds

    if args.lmbda_tune:
        lmbdas = np.arange(0.1,10,0.3).tolist()
    else:
        lmbdas = [args.lmbda]

    if args.lmbda_tune == True:
        elapsetimes = []
        bestnmi = -1
        bestacc = -1
        t = len(lmbdas)
        trivial = [0]*t # Take count on any missing cluster

        for count,lmbda in enumerate(lmbdas):
            print('Inside Lambda ',lmbda)
            print('Inside Sigma ',sigma)

            if N<=5000:
                _,l,elapsed,mode_index,z,_,ts = SLK_iterative(X, sigma, K, W, bound_, SLK_option, C_init,
                                                                   bound_lambda = lmbda, bound_iterations=bound_it)
            else:
                _,l,elapsed,mode_index,z,_,ts = SLK(X, sigma, K, W, bound_, SLK_option, C_init,
                                                         bound_lambda = lmbda, bound_iterations=bound_it)

            if ts:
                trivial[count] = 1
                continue

            # Evaluate the performance on validation set
            current_nmi = nmi(gnd_val,l[val_ind])
            acc,_ = get_accuracy(gnd_val,l[val_ind])

            print('lambda = ',lmbda, ' : NMI= %0.4f' %current_nmi)
            print('accuracy %0.4f' %acc)

            if current_nmi>bestnmi:
                bestnmi = current_nmi
                best_lambda_nmi = lmbda

            if acc>bestacc:
                bestacc = acc
                best_lambda_acc = lmbda

            print('Best result: NMI= %0.4f' %bestnmi,'|NMI lambda = ', best_lambda_nmi)
            print('Best Accuracy %0.4f' %bestacc,'|Acc lambda = ', best_lambda_acc)
            elapsetimes.append(elapsed)

        avgelapsed = sum(elapsetimes)/len(elapsetimes)
        print ('avg elapsed ',avgelapsed)
    else:
        best_lambda_acc = args.lmbda

    ### Run with best Lambda and assess accuracy over whole dataset
    best_lambda = best_lambda_acc # or best_lambda_nmi
    if N>=5000:
        C,l,elapsed,mode_index,z,_,_ = SLK(X,sigma,K,W,bound_,SLK_option,C_init,
                                                bound_lambda = best_lambda, bound_iterations=bound_it)
    else:
        C,l,elapsed,mode_index,z,_,_ = SLK_iterative(X,sigma,K,W,bound_,SLK_option,C_init,
                                                      bound_lambda = best_lambda, bound_iterations=bound_it)
    # Evaluate the performance on dataset

    print('Elapsed time for SLK = %0.5f seconds' %elapsed)
    nmi_ = nmi(gnd_labels,l)
    acc_,_ = get_accuracy(gnd_labels,l)

    print('Result: NMI= %0.4f' %nmi_)
    print('        Accuracy %0.4f' %acc_)
    best_lambda = best_lambda_acc

    if saveresult:
        saveresult_path = data_dir + 'Result_'+dataset+'.mat'
        sio.savemat(saveresult_path,{'lmbda':best_lambda,'l':l,'C':C,'z':z})

    if mode_images and X_org is not None:
        if SLK_option == 'BO':
            mode_images_path = data_dir+dataset+'_modes'
            original_image_size = (28,28)
            util.mode_nn(mode_index,X,K,C,l,6,X_org,mode_images_path, original_image_size)
        else:
            print('\n For Mode images change option to -- BO and have image intensities X_org')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SLK clustering")
    parser.add_argument('--seed', type=int, default=None)
    # dataset
    parser.add_argument('-d', '--dataset', type=str, default='mnist_gan',
                        choices=dataset_names())
    # clustering method
    parser.add_argument('--SLK_option', type=str, default='MS')
    # parser.add_argument('--knn', type=int, default=5)

    parser.add_argument('--lmbda', type=float, default=1.3) # specified lambda
    parser.add_argument('--lmbda-tune', type=util.str2bool, default=False)  # run in a range of different lambdas
    parser.add_argument('--bound', type=util.str2bool, default=True)  # Bound optimization
    parser.add_argument('--mode-images', type=util.str2bool, default=False)  # run in a range of different lambdas
    parser.add_argument('--saveresult', type=util.str2bool, default=False)  # save results


    main(parser.parse_args())
