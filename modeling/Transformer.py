
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class TSTransformerEncoder(nn.Module):

    def __init__(self, input_dimension, output_dimension, hidden_dimmension,
                 attention_heads, encoder_number_of_layers,  
                 positional_encodings, dropout, dim_feedforward=512, activation='gelu'):
        super(TSTransformerEncoder, self).__init__()

        self.project_input = nn.Linear(input_dimension, hidden_dimmension)

        self.hidden_dimmension = hidden_dimmension
        if attention_heads is None:
            attention_heads=hidden_dimmension//64
        self.attention_heads = attention_heads
        self.positional_encodings = positional_encodings

        self.encoder = nn.Linear(input_dimension, hidden_dimmension) # using linear projection instead
        self.pos_encoder = PositionalEncoding(hidden_dimmension, dropout)
        print(hidden_dimmension, attention_heads)
        encoder_layer = TransformerEncoderLayer(hidden_dimmension, self.attention_heads, dim_feedforward, dropout, activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, encoder_number_of_layers)

        self.output_layer = nn.Linear(hidden_dimmension, output_dimension)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.input_dimension = output_dimension

    def forward(self, src,trg_):#, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        src = self.project_input(src)                                                     # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        if self.positional_encodings:                                                   # add positional encoding
            src = self.pos_encoder(src)
                                                         
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # output = self.transformer_encoder(src, src_key_padding_mask=~padding_masks)     # (seq_length, batch_size, d_model)
        output = self.transformer_encoder(src)     # (seq_length, batch_size, d_model)
        output = self.act(output)                                                       # the output transformer encoder/decoder embeddings don't include non-linearity
        # output = output.permute(1, 0, 2)                                                # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
        return output




class TransformerModel(nn.Module):
    def __init__(self, input_dimension, output_dimension, 
                        hidden_dimmension, attention_heads, 
                        encoder_number_of_layers, decoder_number_of_layers, dropout, positional_encodings):
        super(TransformerModel, self).__init__()
        if not attention_heads:
            attention_heads = hidden_dimmension//64
        
        # self.encoder = nn.Embedding(intoken, hidden)
        self.encoder = nn.Linear(input_dimension, hidden_dimmension) # using linear projection instead
        self.pos_encoder = PositionalEncoding(hidden_dimmension, dropout)

        # self.decoder = nn.Embedding(outtoken, hidden)
        self.decoder = nn.Linear(output_dimension, hidden_dimmension) # using linear projection instead
        self.pos_decoder = PositionalEncoding(hidden_dimmension, dropout)

        self.transformer = nn.Transformer(d_model=hidden_dimmension, 
                                          nhead=attention_heads, 
                                          num_encoder_layers=encoder_number_of_layers, 
                                          num_decoder_layers=decoder_number_of_layers, 
                                          dim_feedforward=hidden_dimmension*4, 
                                          dropout=dropout, activation='relu')
        self.fc_out = nn.Linear(hidden_dimmension, output_dimension)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None
        self.positional_encodings = positional_encodings

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        src_pad_mask = self.make_len_mask(src).squeeze(-1)
        trg_pad_mask = self.make_len_mask(trg).squeeze(-1) # [batch size, length]


        src = self.encoder(src)
        if self.positional_encodings:
            src = self.pos_encoder(src)

        trg = self.decoder(trg)
        if self.positional_encodings:
            trg = self.pos_decoder(trg)

        output = self.transformer(src, trg)
        # output = self.transformer(src, trg, 
        #                           src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
        #                           src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)

        return output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=501):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)