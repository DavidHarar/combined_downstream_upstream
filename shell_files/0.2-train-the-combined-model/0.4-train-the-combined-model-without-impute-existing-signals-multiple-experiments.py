
import os
os.chdir('/home/david/Desktop/projects/thesis/')

print(os.getcwd())
print(os.listdir())

import sys
sys.path.append('/home/david/Desktop/projects/thesis/')

import torch

# hyperparams to be changed if needed
downstream_path = './downstream_classification/models/3.0-inception-bs128-balanced-death-lr0.001-decay0-aucpr-weighted33.pt'
upstream_path = './upstream_seq2seq/models/transformer_cnn_1696038569.0264485.pt'

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
# ---------------

from combined_downstream_upstram.executors.train_on_local_machine_mps import run_experiments, trainer

constants = {
    'seed':42,
    'metadata_file_path': './downstream_classification/data/combined_data/metadata_only_existant_readings_09042023.csv',
    'data_folder_path': './downstream_classification/data/individual-signals/',
    'fillna': 0,
    'n_epochs':60,
    'batch_size':64, 
    'verbosity': False,
    'patience':5,
    'clip':1,
    'impute_only_missing': True,
    'minimal_number_of_leads': None,
    'leads': ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'],
    'eval_metric': 'aucpr',
    'minimal_number_of_leads': None,
}
variables = {
    'weight_decay': [0,0.0001,0.0005,0.001],
    'lr': [0.0005,0.001,0.003],
    'loss_function_weight': [33,50,66,100],
    'targets': [['one_year_until_death'],['DM'],['AF']]
}

other_vars = {
        'notebook': '0.4',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

run_experiments(other_vars, constants, variables)



