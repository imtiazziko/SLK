# SLK

This is the code for Scalable Laplacian K-modes (SLK) algorithm for large scale data clustering. If you use this code please cite the following NIPS 2018 paper:

[**Scalable Laplacian K-modes**](https://papers.nips.cc/paper/8208-scalable-laplacian-k-modes.pdf)  
Imtiaz Masud Ziko, Eric Granger and Ismail Ben Ayed  
In *Neural Information Processing Systems conference (NEURIPS)*, Montreal, Canada, December 2018.

## Prerequisites

The code is run and tested with python 3.X and need the following packages:

- pyflann
- annoy (if used as an option)

## Usage

We give the the example script [test_SLK.py](test_SLK.py) for MNIST dataset with Feature learned from running simple MLP GAN network. The learned features is in [gan_mnist.mat](gan_mnist.mat) in data folder.  

To test simply run the following which run with the tuned lambda and initial seed which get around 94% accuracy on MNIST 
```
python test_SLK.py
```

- To save the mode images and the clustering results set the parameters in the line 26 and 27 to True
- To run options SLK-BO or SLK-MS change SLK_option in [test_SLK.py](test_SLK.py)
- To test with other datasets give the dataset as a feature array of form number of samples (N) x number of dimension (D) or a memmap file or accordingly.

## Modes
Example modes found with SLK for MNIST

<div align="center"><img src="data/mnist_mode_mean.png" alt="" height="400" width="500"/></div>


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

