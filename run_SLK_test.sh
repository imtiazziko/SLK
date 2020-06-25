#!/bin/bash


bound_=True # bound update
tune=False  # Tune Lambda
lmd=1.31  # Given Lambda
mode=BO    # With modes from Bound updates
# mode=MS   # With mean-shift modes
#### Demo for Mnist with Gan features
dataset=mnist_gan
python main_SLK.py -d $dataset \
                       --SLK_option $mode \
                       --lmbda $lmd \
                       --lmbda-tune $tune \
                       --bound $bound_

## Other datasets. Download the datasets first before uncommenting the followings
#dataset=mnist
#python main_SLK.py -d $dataset \
#                       --SLK_option $mode \
#                       --lmbda $lmd \
#                       --lmbda-tune $tune \
#                       --bound $bound_
##
#python main_SLK.py -d $dataset \
#                       --SLK_option $mode \
#                       --lmbda $lmd \
#                       --lmbda-tune $tune \
#                       --bound $bound_

#mode=Means
#dataset=mnist_code
#python main_SLK.py -d $dataset \
#                       --SLK_option $mode \
#                       --lmbda $lmd \
#                       --lmbda-tune $tune \
#                       --bound $bound_
#
##dataset=labelme_alex
##python main_SLK.py -d $dataset \
##                       --SLK_option $mode \
##                       --lmbda $lmd \
##                       --lmbda-tune $tune \
##                       --bound $bound_
#dataset=labelme_gist
#python main_SLK.py -d $dataset \
#                       --SLK_option $mode \
#                       --lmbda $lmd \
#                       --lmbda-tune $tune \
#                       --bound $bound_
#dataset=shuttle
#python main_SLK.py -d $dataset \
#                       --SLK_option $mode \
#                       --lmbda $lmd \
#                       --lmbda-tune $tune \
#                       --bound $bound_
##dataset=ytf
##python main_SLK.py -d $dataset \
##                       --SLK_option $mode \
##                       --lmbda $lmd \
##                       --lmbda-tune $tune \
##                       --bound $bound_
#
#dataset=reuters
##mode=Means
##python main_SLK.py -d $dataset \
##                       --SLK_option $mode \
##                       --lmbda $lmd \
##                       --lmbda-tune $tune \
##                       --bound $bound_
#mode=MS
#python main_SLK.py -d $dataset \
#                       --SLK_option $mode \
#                       --lmbda $lmd \
#                       --lmbda-tune $tune \
#                       --bound $bound_
#
#mode=BO
#python main_SLK.py -d $dataset \
#                       --SLK_option $mode \
#                       --lmbda $lmd \
#                       --lmbda-tune $tune \
#                       --bound $bound_