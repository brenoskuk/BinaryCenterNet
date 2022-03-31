# Installation

## Installing requirements 

The code was tested on Ubuntu 20.04, with [Anaconda](https://www.anaconda.com/download) Python 3.8.5, Tensorflow 2.4.1, larq 0.12.0

After install Anaconda:

0. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name BinaryCenterNet python=3.8.5
    ~~~
    And activate the environment.
    
    ~~~
    conda activate BinaryCenterNet
    ~~~

1. Install Tensorflow:

    ~~~
    conda install Tensorflow=2.4.1
    ~~~
    

2. Install Larq:

    ~~~
    pip install larq=0.12.2
    ~~~  

3. Install cython:

    ~~~
    conda install -c anaconda cython=0.29.21
    ~~~


5. [Optional] Install Larq Zoo:

    ~~~
    pip install larq=2.1.0
    ~~~  
     

## Alternative: Using requirements.yml

~~~
conda env create -f requirements.yml
~~~
     
If sucessfull an environment called BinaryCenterNet will be created containing all of the requirements to run the project.

# [Optional] Sanity check with a tiny dataset

If you want to check if the installation is working before downloading the whole PascalVOC dataset you can train and evaluate a toy architecture on a tiny subsample of PascalVOC2007. Run the following command to train a model on only two images.

~~~
python train.py --architecture custom_qn_centernet --dataset-path PascalVOC-tiny-subset --batch_size 2 --epochs 2 --val-sampling 1 --train-sampling 1 --no-snapshots 
~~~

A model summary should be printed and the model should train given the arguments. Note that evaluation is performed just to check the pipeline. For more information on training models please refer to README.md

# Downloading PascalVOC dataset 

The train and test set used in the project uses the PascalVOC2007 and the PascalVOC2012 datasets. The merging of these sets has become a standard for evaluation and training of published Object Detection papers.

We make available a repository containing the already merged data. To download it clone it using the following git command:


~~~
git clone https://github.com/brenoskuk/PascalVOC-OD-2007-2012.git
~~~

The dataset contains:

* A train set that consists of: Pascal VOC2007 trainval merged with Pascal VOC2012 trainval
* A test set that consists of: Pascal VOC2007 test 

After downloading it place the folder "
PascalVOC-OD-2007-2012" inside the folder "datasets". 