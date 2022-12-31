import os, sys, glob
import numpy as np
import torch.nn as nn
from torch import Tensor

class STRGNN(nn.Module):
    def __init__(self, encoder, decoder):
        super(STRGNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder