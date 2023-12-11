
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import torch

from tqdm import tqdm

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from upstream_seq2seq.modeling.Transformer import *
from downstream_classification.modeling.Inception import *



class CombinedModel(torch.nn.Module):
    def __init__(self, upstream_model, downstream_model, device, continue_training_upstream_model, impute_only_missing=False):
        super(CombinedModel, self).__init__()
        
        self.upstream_model                     = upstream_model
        self.downstream_model                   = downstream_model
        self.impute_only_missing                = impute_only_missing
        self.continue_training_upstream_model   = continue_training_upstream_model
        self.device                             = device
        
        if self.continue_training_upstream_model:
            self.upstream_model.train()
        else:
            self.upstream_model.eval()

    def forward(self, x):
        # Should we calculate the gradients for the upstream task?
        if not self.continue_training_upstream_model:
            with torch.no_grad():
                Y = self.upstream_model(x,None)
        else:
            Y = self.upstream_model(x,None)
        Y = Y.permute((1,2,0))        
        # if impute only missing signals
        if self.impute_only_missing:
            Y = CombinedModel.X_imputer(X=x.permute((1,2,0)),Y=Y, device=self.device)
        Y = self.downstream_model(Y)
        return Y
    
    
    @staticmethod
    def get_non_zero_leads(X):
        means = np.mean(X,axis=-1) != 0
        return means
    
    @staticmethod
    def X_imputer(X,Y, device):
        X_ = X.detach().cpu().numpy()
        Y_ = Y.detach().cpu().numpy()
        non_zeros_mask = CombinedModel.get_non_zero_leads(X_)
        # X_[non_zeros_mask] = X_[non_zeros_mask]
        X_[~non_zeros_mask] = Y_[~non_zeros_mask]
        
        # sanity check
        # print('imputed')
        # print('all:', np.mean(X_ == Y_))
        # print('mask:', np.mean(X_[non_zeros_mask] == Y_[non_zeros_mask]))
        # print('not mask:',np.mean(X_[~non_zeros_mask] == Y_[~non_zeros_mask]))
        
        X_ = torch.from_numpy(X_).to(device)
        return X_

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# combined_model = CombinedModel()
# print(combined_model)