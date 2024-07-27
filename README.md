# STRGNN: Sequentially Topological Regularization Graph Neural Network
This repository contains the implementation and the datasets for our paper "Deep learning of multimodal networks with topological regularization for drug repositioning".

STRGNN effectively predicts the association between drugs and diseases based on large-scale multimodal networks containing abundant omics information.

We proposed "topological regularization", which appropriately selects informative modalities and discard redundant ones from the multimodal network data.

<div align="center">
  <img src="assets/overview.png" width="80%">
</div>

# Requirements
STRGNN is tested to work under Python 3.8.
Below packages are required by STRGNN.
  - torch (1.4.0)
  - numpy (1.19.0)
  - torch-cluster (1.5.4)
  - torch-scatter (2.0.4)
  - torch-sparse (0.6.1)
  - torch-spline-conv (1.2.0)
  - torch-geometric (1.6.0)

All the required packages can be installed using the following command:

```
$ pip install -r requirements.txt
```

# Quick start
1. Unzip multimodal network dataset.

```
$ cd Dataset
$ python unzip.py
```

2. Run STRGNN as following command:

```
$ cd src
$ python main.py
```

Options are: <br>
```
-c: The number of cycle, default=2
-d: The embedding dimension, default=128
-e: The number of epochs, default=5000
-g: The cuda devive, default=0
-s: The stride size, default=2
-v: The number of verbose, default=10
-lr: The learning rate, default=0.001
-cs: The channel size, default=5
-cv: The number of cross-validation, default=1
-de: The drop edge ratio, default=0.2
-do: The dropout ratio, default=0.5
-ks: The kernel size, default=3
-ls: The embedding loss, default='none'
-nl: The parameter of normalization lambda, default=1e-6
-rt: The regularization type, default='l1'
-tg: The target network, default='drug_disease'
-dec: The decoder type, default='IPD'
-enc: The encoder type, default='MIX'
-save: The parameter to decide save model, default=False
-type: The link prediction task type, default='transductive'
-sno: (Option) Shuffle network order, default='none'
-nop: (Oprion) Randomly insert/remove input networks, default='none'
-nrr: (Option) The ratio of input networks insertion/removal, default='none'
```


# Citing
If you use STRGNN or our dataset, please cite our paper.

# Contacts
Please contact me at yuuto.0902@dna.bio.keio.ac.jp for any questions or comments.