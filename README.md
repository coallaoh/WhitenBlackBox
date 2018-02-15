# Towards Reverse-Engineering Black-Box Neural Networks, ICLR'18

#### Seong Joon Oh, Max Augustin, Bernt Schiele, Mario Fritz.

#### Max-Planck Institute for Informatics.

[Towards Reverse-Engineering Black-Box Neural Networks](https://arxiv.org/abs/1711.01768), ICLR'18

Many deployed learned models are black boxes: given input, returns output. Internal information about the model, such as the architecture, optimisation procedure, or training data, is not disclosed explicitly as it might contain proprietary information or make the system more vulnerable. This work shows that such attributes of neural networks can be exposed from a sequence of queries. This has multiple implications. On the one hand, our work exposes the vulnerability of black-box neural networks to different types of attacks -- we show that the revealed internal information helps generate more effective adversarial examples against the black box model. On the other hand, this technique can be used for better protection of private content from automatic recognition models using adversarial examples. Our paper suggests that it is actually hard to draw a line between white box and black box models.

## Metamodels for reverse-engineering network details

We extract diverse types of information from a black-box neural network (which we call _model attributes_; examples include the non-linear activation type, optimisation algorithm, training dataset) by observing its output with respect to certain query inputs. This is achieved by learning the correlation between the network attributes and certain patterns in the network's output patterns. The correlation is learned by training a classifier over outputs from multiple models to predict the attributes - we call this a _metamodel_ because it literally classifies classifiers. We introduce three novel metamodel methods in this paper. They differ in the way they choose the query inputs and interpret the corresponding outputs.

### kennen-o

<p align="center">
<img align="middle" src="http://datasets.d2.mpi-inf.mpg.de/joon18iclr/meta-arch-or.png" width="500" >
</p>

The simplest one - `kennen-o` - selects the query inputs at random from a dataset. An MLP classifier is trained over the outputs with respect to the selected inputs to predict network attributes. See the figure above.

### kennen-i

<p align="center">
<img align="middle" src="http://datasets.d2.mpi-inf.mpg.de/joon18iclr/meta-arch-ic.png" width="500" >
</p>

Our second approach - `kennen-i` - approaches the problem from a completely different point of view. For the sake of clarity, we take an MNIST digit classifier as an example. Over multiple white-box models (training set models), we craft an _input_ that is designed to expose inner secrets of the training set models. This crafted input turns out to generalise very well to unseen black-box models, in the sense that it also reveals the secrets of the unseen black box. More specifically, using gradient signals from a diverse set of white box models, we design a query input that forces an MNIST digit classifier to predict `0` if the classifier has the attribute A, and `1` if it doesn't. In other words, the crafted input _re-purposes_ a digit classifier into a model attribute classifier. See the figure above for the training procedure. We also show below some learned query inputs which are designed to induce the prediction of label `0` if the victim black box has a max-pooling layer, train-time dropout layer, and kernel size 3, respectively, and `1` otherwise.

|| Max-Pooling, yes or no? | Dropout, yes or no? | Kernel Size, 3 or 5?  |
| --- | :---: | :---: | :---: |
|Crafted input| <img src="http://datasets.d2.mpi-inf.mpg.de/joon18iclr/pool.jpg" width="100">  | <img src="http://datasets.d2.mpi-inf.mpg.de/joon18iclr/drop.jpg" width="100"> | <img src="http://datasets.d2.mpi-inf.mpg.de/joon18iclr/ks.jpg" width="100"> |
|Reverse-engineering <br> success rate <br> (random chance)| 94.8% <br> (50%) | 77.0% <br> (50%) | 88.5% <br> (50%) |

They share similarities to adversarial examples to neural networks ([Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)) that are also designed to alter the behaviour a neural network. The only difference is the goal. The goal of adversarial examples is to induce a specific output (e.g. wrong output, specific prediction for malicious purpose). The goal of the `kennen-i` inputs is to _expose_ the model attributes. They both seem to generalise well to unseen models, enabling attacks on black boxes. (See [Delving into Transferable Adversarial Examples and Black-Box Attacks](https://arxiv.org/abs/1611.02770) for transferability of adversarial examples.)

### kennen-io

Our final metamodel - `kennen-io` - combines `kennen-o` and `kennen-i`. For detailed experimental results on MNIST and ImageNet classifiers, see the paper!

## Environment

We only support python 2 for this project. Conda environment with [pytorch](http://pytorch.org/) (with cuda 8.0) has been used. 

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

## (Optional) Download the *MNIST-NET* dataset

*MNIST-NET* is a dataset of 11,282 diverse MNIST digit classifiers. The full pipeline for generating *MNIST-NET* is included in the repository (see below). The generation has taken about 40 GPU days with NVIDIA Tesla K80. Alternatively, the dataset can be downloaded from [this link](https://datasets.d2.mpi-inf.mpg.de/joon18iclr/MNIST-NET.tar.gz) (19GB). Untar the downloaded file in the `cache/` folder. 

## Running the code

Running 
```bash
$ ./run.py
```
will (1) generate the *MNIST-NET* dataset and (2) train and evaluate various metamodels (kennen variants - see paper) on the *MNIST-NET*. Read `run.py` in detail for more information on configuration etc.

## Contact

For any problem with implementation or bug, please contact [Seong Joon Oh](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/seong-joon-oh/) (joon at mpi-inf dot mpg dot de).

## Citation

```
  @article{joon18iclr,
    title = {Towards Reverse-Engineering Black-Box Neural Networks},
    author = {Oh, Seong Joon and Augustin, Max and Schiele, Bernt and Fritz, Mario},
    year = {2018},
    journal = {International Conference on Learning Representations},
  }
```
