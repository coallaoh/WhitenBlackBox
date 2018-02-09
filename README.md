# Whitening Black-Box Neural Networks, ICLR'18

#### Seong Joon Oh, Max Augustin, Bernt Schiele, Mario Fritz.

#### Max-Planck Institute for Informatics.

[Whitening Black-Box Neural Networks](https://arxiv.org/abs/1711.01768), ICLR'18

Many deployed learned models are black boxes: given input, returns output. Internal information about the model, such as the architecture, optimisation procedure, or training data, is not disclosed explicitly as it might contain proprietary information or make the system more vulnerable. This work shows that such attributes of neural networks can be exposed from a sequence of queries. This has multiple implications. On the one hand, our work exposes the vulnerability of black-box neural networks to different types of attacks -- we show that the revealed internal information helps generate more effective adversarial examples against the black box model. On the other hand, this technique can be used for better protection of private content from automatic recognition models using adversarial examples. Our paper suggests that it is actually hard to draw a line between white box and black box models.

## Environment

We only support python 2 for this project. Conda environment with pytorch (with cuda 8.0) has been used. 

## Installation

Clone this repository recursively.

```bash
$ git clone https://github.com/coallaoh/WhitenBlackBox.git --recursive
```

## Download data

Run the following commands to download and untar the necessary data (6.3MB).

```bash
$ mkdir cache && wget https://datasets.d2.mpi-inf.mpg.de/joon18iclr/mnist_val.pkl.tar.gz -P cache/ && cd cache && tar xvf mnist_val.pkl.tar.gz && cd ..
```

## (Optional) Download *MNIST-NET* dataset

*MNIST-NET* is a dataset of 11,282 diverse MNIST digit classifiers. The full pipeline for generating *MNIST-NET* is included in the repository (see below). The generation has taken about 40 GPU days with NVIDIA Tesla K80. Alternatively, the dataset can be downloaded from [this link](https://datasets.d2.mpi-inf.mpg.de/joon18iclr/MNIST-NET.tar.gz) (19GB). Untar the downloaded file in the `cache/` folder. 

## Running the code

Running 
```bash
$ ./run.py
```
will (1) generate the *MNIST-NET* dataset and (2) train and evaluate various metamodels (kennen variants - see paper) on the *MNIST-NET*. Read `run.py` in detail for more information on configuration etc.



