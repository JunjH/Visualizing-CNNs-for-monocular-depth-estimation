from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import pdb
import copy
import numpy as np
import modules
from torchvision import utils

import senet
import resnet
import densenet
import net_mask

class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R(block_channel)


    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out

class model_2(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model_2, self).__init__()

        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R(block_channel)

        original_model2 = net_mask.drn_d_22(pretrained=True)
        self.mask = net_mask.AutoED(original_model2)  


    def forward(self, x):
        mask = self.mask(x)
        x_block1, x_block2, x_block3, x_block4 = self.E(x*mask)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))


        return out,mask

class model_3(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model_3, self).__init__()

        self.E = Encoder
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R2()

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[114,152])
        out = self.R(x_mff)

        return out

class model_4(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model_4, self).__init__()

        self.E = Encoder
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R2()

        original_model2 = net_mask.drn_d_22(pretrained=True)
        self.mask = net_mask.AutoED(original_model2)  

    def forward(self, x):
        mask = self.mask(x)
        x_block1, x_block2, x_block3, x_block4 = self.E(x*mask)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[114,152])
        out = self.R(x_mff)

        return out,mask



class model2(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model2, self).__init__()

        self.E = Encoder
        self.D = modules.D2(num_features)



    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        out = self.D(x_block1, x_block2, x_block3, x_block4)

        return out

class model3(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model3, self).__init__()

        self.E = Encoder
        self.D = modules.MFF2(num_features)


    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        out = self.D(x_block1, x_block2, x_block3, x_block4)

        return out