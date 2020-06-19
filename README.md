# SLK

This is the code for Scalable Laplacian K-modes (SLK) algorithm for large scale data clustering. If you use this code please cite the following NEURIPS 2018 paper:

[**Scalable Laplacian K-modes**](https://papers.nips.cc/paper/8208-scalable-laplacian-k-modes.pdf)  
Imtiaz Masud Ziko, Eric Granger and Ismail Ben Ayed  
In *Neural Information Processing Systems conference (NEURIPS)*, Montreal, Canada, December 2018.

## Prerequisites

1. The code is tested on python 3.6. Install the requirements listed in ([requirements.txt](./requirements.txt)) using pip or conda.
2. Download the datasets features from [google drive link](https://drive.google.com/file/d/1cOcP7_gZPNk_m5N8soV2Wqb4Adcs8WIi/view?usp=sharing).

## Usage
To evaluate the code simply run the following script: 
```
sh run_SLK_test.sh
```
Change the options inside the scripts accordingly. The options are fairly described in the script and in ([main_SLK.py](./main_SLK.py))

## Example modes
Example modes found with SLK for MNIST

<div align="center"><img src="data/mnist_mode_mean.png" alt="" height="400" width="500"/></div>


<!-- ## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. -->

