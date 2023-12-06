import os
os.chdir('/home/david/Desktop/projects/thesis/')

import torch

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
# ---------------


from combined_downstream_upstram.executors.train_on_local_machine_mps import trainer


config = {
    # general
    'seed': 123,
    'metadata_file_path': './downstream_classification/data/combined_data/metadata_only_existant_readings_09042023.csv',
    'data_folder_path': './downstream_classification/data/individual-signals/',
    'fillna': -1,
    # training
    'batch_size': 64,
    'n_epochs': 60,
    'clip':1,
    'targets': ['one_year_until_death'],
    'saving_path': './combined_downstream_upstram/models/comined_model_training',
}

import os

# run
print('Starting Experiment')
trainer(**config)
