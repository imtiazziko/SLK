
import os.path as osp
import scipy.io as sio
import numpy as np


__datasets = ['mnist_gan', 'mnist', 'mnist_code', 'labelme_alex', 'labelme_gist', 'shuttle', 'ytf', 'reuters']

def dataset_names():

    return __datasets


def read_dataset(name, data_dir):

    X, gnd_labels, K, sigma, X_org = None, None, None, None, None

    if name not in __datasets:
        raise KeyError("Dataset not implemented:",name)

    elif name == 'mnist_gan':
        data = np.load(data_dir+name+'.npz')
        X = data['X']                               # data features
        gnd_labels = data['gnd_labels'] # Ground truth for evaluation
        K = 10
        sigma = 0.4
        X_org = np.load(data_dir+'mnist.npz')['X']*255.0 # Original image intensities for mode visualization
        knn=5
    elif name == 'mnist_code':
        data = np.load(data_dir+name+'.npz')
        X = data['X']                               # data features
        gnd_labels = data['gnd_labels']  # Ground truth for evaluation
        K = 10
        sigma = 0.4
        X_org = np.load(data_dir+'mnist.npz')['X']*255.0 # Original image intensities for mode visualization
        knn=5
    elif name == 'mnist':
        data = np.load(data_dir+name+'.npz')
        X = data['X']                               # data features
        gnd_labels = data['gnd_labels']  # Ground truth for evaluation
        K = 10
        sigma = 0.4
        X_org = X*255.0
        knn=5
    elif name == 'labelme_alex':
        data = np.load(data_dir+name+'.npz')
        X = data['X']                               # data features
        gnd_labels = data['gnd_labels']  # Ground truth for evaluation
        K = 8
        sigma = None
        X_org = None
        knn=5
    elif name == 'labelme_gist':
        data = np.load(data_dir+name+'.npz')
        X = data['X']                               # data features
        gnd_labels = data['gnd_labels']  # Ground truth for evaluation
        K = 8
        sigma = None
        X_org = None
        knn=5

    elif name == 'shuttle':
        data = np.load(data_dir+name+'.npz')
        X = data['X']                               # data features
        gnd_labels = data['gnd_labels']  # Ground truth for evaluation
        K = 7
        sigma = None
        X_org = None
        knn=10

    elif name == 'ytf':
        data = np.load(data_dir+name+'.npz')
        X = data['X']                               # data features
        gnd_labels = data['gnd_labels']  # Ground truth for evaluation
        K = 40
        sigma = None
        X_org = None
        knn=5

    elif name == 'reuters':
        X = np.load(data_dir+'reuters_X.npy')                              # data features
        gnd_labels = np.load(data_dir+'reuters_gnd_labels.npy')  # Ground truth for evaluation
        K = 4
        sigma = None
        X_org = None
        knn=10

    else:
        pass

    # if X.shape[0]> 2000:
    #     knn=10
    # else

    return X, gnd_labels, K, sigma, X_org, knn


if __name__=='__main__':


    dataset = 'labelme_gist'
    data_dir = '../data/'
    X, gnd_labels, K, sigma, X_org = read_dataset(dataset, data_dir)