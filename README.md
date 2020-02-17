# Ladder Variational Autoencoders (LVAE)

PyTorch implementation of Ladder Variational Autoencoders (LVAE) [1]:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![LVAE equation](_readme_figs/lvae_eq.png)

where the variational distributions _q_ at each layer are multivariate Normal
with diagonal covariance.

**Significant differences from [1]** include:
- skip connections in the generative path: conditioning on _all_ layers above
rather than only on _the_ layer above (see for example [2])
- spatial (convolutional) latent variables
- free bits [3] instead of beta annealing [4]

### Install requirements and run MNIST example

```
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python main.py --zdims 32 32 32 --downsample 1 1 1 --nonlin elu --skip --blocks-per-layer 4 --gated --freebits 0.5 --learn-top-prior --data-dep-init --seed 42 --dataset static_mnist
```

Dependencies include [boilr](https://github.com/addtt/boiler-pytorch) (a framework 
for PyTorch) and [multiobject](https://github.com/addtt/multi-object-datasets)
(which provides multi-object datasets with PyTorch dataloaders).



## Likelihood results

Log likelihood bounds on the test set. Final results coming soon.

|  dataset             | num layers | -ELBO        | - log _p(x)_ ≤ <br> [100 iws] | - log _p(x)_ ≤ <br> [1000 iws] |
| -------------------- |:----------:|:------------:|:-------------:|:--------------:|
| binarized MNIST      | 3          | 82.14        | 79.47         | 79.24          |
| binarized MNIST      | 6          | 80.74        | 78.65         | 78.52          |
| binarized MNIST      | 12         | 80.50        | 78.50         | 78.30          |
| multi-dSprites (0-2) | 12         | 26.9         | 23.2          |     |
| SVHN                 | 15         | 4012 (1.88)  | 3973 (1.87)   |     |
| CIFAR10              | 3          | 7651 (3.59)  | 7591 (3.56)   |     |
| CIFAR10              | 6          | 7321 (3.44)  | 7268 (3.41)   |     |
| CIFAR10              | 15         | 7128 (3.35)  | 7068 (3.32)   |     |
| CelebA               | 20         | 20026 (2.35) | 19913 (2.34)  |     |

Note:
- Bits per dimension in brackets.
- 'iws' stands for importance weighted samples. More samples means tighter log
  likelihood lower bound. The bound converges to the actual log likelihood as 
  the number of samples goes to infinity [5].
- Each pixel in the images is modeled independently. The likelihood is Bernoulli
  for binary images, and discretized mixture of logistics with 10 
  components [6] otherwise.


## Supported datasets

- Statically binarized MNIST [7], see Hugo Larochelle's website `http://www.cs.toronto.edu/~larocheh/public/datasets/`
- [SVHN](http://ufldl.stanford.edu/housenumbers/)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) rescaled and cropped to 64x64 &ndash; see code for details. The path in `experiment.data.DatasetLoader` has to be modified
- [binary multi-dSprites](https://github.com/addtt/multi-object-datasets): 64x64 RGB shapes (0 to 2) in each image


## Samples

#### Binarized MNIST

![MNIST samples](_readme_figs/mnist_sample_1.png)

#### Multi-dSprites

![multi-dSprites samples](_readme_figs/multidsprites_sample_1_downsc.png)

#### SVHN

![SVHN samples](_readme_figs/svhn_sample_1.png)

#### CIFAR

![CIFAR samples](_readme_figs/cifar_sample_1.png)

#### CelebA

![CelebA samples](_readme_figs/celeba_sample_1_downsc.png)


## Hierarchical representations

Here we try to visualize the representations learned by individual layers.
We can get a rough idea of what's going on at layer _i_ as follows:

- Sample latent variables from all layers above layer _i_ (Eq. 1).

- With these variables fixed, take _S_ conditional samples at layer _i_ (Eq. 2). Note
  that they are all conditioned on the same samples. These correspond to one 
  row in the images below.

- For each of these samples (each small image in the images below), pick the 
  mode/mean of the conditional distribution of each layer below (Eq. 3).
  
- Finally, sample an image ***x*** given the latent variables (Eq. 4).

Formally:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![](_readme_figs/eq_layer_inspection.png)

where _s_ = 1, ..., _S_ denotes the sample index.

The equations above yield _S_ sample images conditioned on the same values of 
***z*** for layers _i_+1 to _L_. These _S_ samples are shown in one row of the
images below.
Notice that samples from each row are almost identical when the variability comes
from a low-level layer, as such layers mostly model local structure and details.
Higher layers on the other hand model global structure, and we observe more and
more variability in each row as we move to higher layers. When the sampling 
happens in the top layer (_i = L_), all samples are completely independent, 
even within a row.

#### Binarized MNIST: layers 4, 8, 10, and 12 (top)

![MNIST layers 4](_readme_figs/layers_mnist/sample_mode_layer3.png)&nbsp;&nbsp;
![MNIST layers 8](_readme_figs/layers_mnist/sample_mode_layer7.png)

![MNIST layers 10](_readme_figs/layers_mnist/sample_mode_layer9.png)&nbsp;&nbsp;
![MNIST layers 12](_readme_figs/layers_mnist/sample_mode_layer11.png)


#### SVHN: layers 4, 10, 13, and 15 (top)

![SVHN layers 4](_readme_figs/layers_svhn/sample_mode_layer3.png)&nbsp;&nbsp;
![SVHN layers 10](_readme_figs/layers_svhn/sample_mode_layer9.png)

![SVHN layers 13](_readme_figs/layers_svhn/sample_mode_layer12.png)&nbsp;&nbsp;
![SVHN layers 15](_readme_figs/layers_svhn/sample_mode_layer14.png)


#### CIFAR: layers 3, 7, 10, and 15 (top)

![CIFAR layers 3](_readme_figs/layers_cifar/sample_mode_layer2.png)&nbsp;&nbsp;
![CIFAR layers 7](_readme_figs/layers_cifar/sample_mode_layer6.png)

![CIFAR layers 10](_readme_figs/layers_cifar/sample_mode_layer9.png)&nbsp;&nbsp;
![CIFAR layers 15](_readme_figs/layers_cifar/sample_mode_layer14.png)


#### CelebA: layers 6, 11, 16, and 20 (top)

![CelebA layers 6](_readme_figs/layers_celeba/sample_mode_layer5.png)

![CelebA layers 11](_readme_figs/layers_celeba/sample_mode_layer10.png)

![CelebA layers 16](_readme_figs/layers_celeba/sample_mode_layer15.png)

![CelebA layers 20](_readme_figs/layers_celeba/sample_mode_layer19.png)


#### Multi-dSprites: layers 3, 7, 10, and 12 (top)

![MNIST layers 4](_readme_figs/layers_multidsprites/sample_mode_layer2.png)&nbsp;&nbsp;
![MNIST layers 8](_readme_figs/layers_multidsprites/sample_mode_layer6.png)

![MNIST layers 10](_readme_figs/layers_multidsprites/sample_mode_layer9.png)&nbsp;&nbsp;
![MNIST layers 12](_readme_figs/layers_multidsprites/sample_mode_layer11.png)



## Experimental details

I did not perform an extensive hyperparameter search, but this worked pretty well:

- Downsampling by a factor of 2 in the beginning of inference.
  After that, activations are downsampled 4 times for 64x64 images (CelebA and 
  multi-dSprites), and 3 times otherwise. The spatial size of the final feature 
  map is always 2x2.
  Between these downsampling steps there is approximately the same number of
  stochastic layers.
- 4 residual blocks between stochastic layers. Haven't tried with more
  than 4 though, as models become quite big and we get diminishing returns.
- The deterministic parts of bottom-up and top-down architecture are (almost)
  perfectly mirrored for simplicity.
- Stochastic layers have spatial random variables, and the number of rvs per
  "location" (i.e. number of channels of the feature map after sampling from a
  layer) is 32 in all layers.
- All other feature maps in deterministic paths have 64 channels.
- Skip connections in the generative model (`--skip`).
- Gated residual blocks (`--gated`).
- Learned prior of the top layer (`--learn-top-prior`).
- A form of data-dependent initialization of weights (`--data-dep-init`).
  See code for details.
- freebits=1.0 in experiments with more than 6 stochastic layers, and 0.5 for
  smaller models.
- For everything else, see `_parse_args()` in `experiment/experiment_manager.py`.

With these settings, the number of parameters is roughly 1M per stochastic
  layer. I tried to control for this by experimenting e.g. with half the number
  of layers but twice the number of residual blocks, but it looks like the number
  of stochastic layers is what matters the most.


## Requirements

Tested with:
```
python 3.7.6
numpy 1.18.1
torch 1.4.0
torchvision 0.5.0
matplotlib 3.1.2
seaborn 0.9.0
boilr 0.6.0
multiobject 0.0.3
```

## References

[1] CK Sønderby,
T Raiko,
L Maaløe,
SK Sønderby,
O Winther.
_Ladder Variational Autoencoders_, NIPS 2016

[2] L Maaløe, M Fraccaro, V Liévin, O Winther.
_BIVA: A Very Deep Hierarchy of Latent Variables for Generative Modeling_,
NeurIPS 2019

[3] DP Kingma,
T Salimans,
R Jozefowicz,
X Chen,
I Sutskever,
M Welling.
_Improved Variational Inference with Inverse Autoregressive Flow_,
NIPS 2016

[4] I Higgins, L Matthey, A Pal, C Burgess, X Glorot, M Botvinick, 
S Mohamed, A Lerchner.
_beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework_,
ICLR 2017

[5] Y Burda, RB Grosse, R Salakhutdinov.
_Importance Weighted Autoencoders_,
ICLR 2016

[6] T Salimans, A Karpathy, X Chen, DP Kingma.
_PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications_,
ICLR 2017

[7] H Larochelle, I Murray.
_The neural autoregressive distribution estimator_,
AISTATS 2011
