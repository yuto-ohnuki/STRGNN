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

    $ pip install -r requirements.txt

# Quick start
1. Unzip multimodal network dataset.

    $ python unzip.py


2. Run STRGNN as following command:

    $ python main.py


Options are: <br>
<code>-c: The number of cycle, default=2</code><br>
<code>-d: The embedding dimension, default=128</code><br>
<code>-e: The number of epochs, default=5000</code><br>
<code>-g: The cuda devive, default=0</code><br>
<code>-s: The stride size, default=2</code><br>
<code>-v: The number of verbose, default=10</code><br>
<code>-cs: The channel size, default=5</code><br>
<code>-cv: The number of cross-validation, default=1</code><br>
<code>-de: The drop edge ratio, default=0.2</code><br>
<code>-do: The dropout ratio, default=0.5</code><br>
<code>-ks: The kernel size, default=3</code><br>
<code>-ls: The embedding loss, default='none'</code><br>
<code>-nl: The parameter of normalization lambda, default=1e-6</code><br>
<code>-rt: The regularization type, default=1</code><br>
<code>-tg: The target network, default='drug_disease'</code><br>
<code>-dec: The decoder type, default='IPD'</code><br>
<code>-enc: The encoder type, default='MIX'</code><br>
<code>-save: The parameter to decide save model, default=False</code><br>
<code>-type: The link prediction task type, default='transductive'</code><br>

# Citing
If you use STRGNN or our dataset, please cite our paper.

# Contacts
Please contact me at yuuto.0902@dna.bio.keio.ac.jp for any questions or comments.