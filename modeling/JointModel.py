
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


# hyperparams to be changed if needed
downstream_path = './downstream_classification/models/3.0-inception-bs128-balanced-death-lr0.001-decay0-aucpr-weighted33.pt'
upstream_path = './upstream_seq2seq/models/transformer_cnn_1696038569.0264485.pt'

lr,dropout,class_weight,weight_decay = (0.001, 0.2, 50.0, 0.0005)

## upstream
input_dimension          = 12
output_dimension         = 12
hidden_dimmension        = 64
attention_heads          = None
encoder_number_of_layers = 8
dim_feedforward          = 512
kernel_size              = 3
activation               ='gelu'
dropout                  = 0.4
device                   = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# loss_function_weight = torch.tensor([class_weight]).to(device)
# # ---------------

# optimizer = optim.AdamW(downstream_model.parameters(), weight_decay=weight_decay, lr=lr)
# criterion = nn.BCELoss(loss_function_weight)




class CombinedModel(torch.nn.Module):
    def __init__(self, upstream_model, downstream_model, impute_only_missing=False):
        super(CombinedModel, self).__init__()
        
        self.upstream_model = upstream_model
        self.downstream_model = downstream_model
        self.impute_only_missing = impute_only_missing
    
    def forward(self, x):

        # with torch.no_grad():
        Y = self.upstream_model(x,None) #.cpu().numpy()
        Y = Y.permute((1,2,0))        
        # if impute only missing signals
        if self.impute_only_missing:
            Y = CombinedModel.X_imputer(X=x.permute((1,2,0)),Y=Y)
        Y = self.downstream_model(Y)
        return Y
    
    
    @staticmethod
    def get_non_zero_leads(X):
        means = np.mean(X,axis=-1) != 0
        return means
    
    @staticmethod
    def X_imputer(X,Y):
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