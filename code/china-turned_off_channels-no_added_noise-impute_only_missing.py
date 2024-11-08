import argparse

import os
import sys

import numpy as np
import pickle 


# import trainer
# -----------------------------
# PC
# sys.path.append('/home/david/Desktop/projects/thesis/downstream_classification')
# os.chdir('/home/david/Desktop/projects/thesis/downstream_classification')

# MAC
# sys.path.append('/Users/davidharar/Documents/School/thesis/git_repositories/')
# os.chdir('/Users/davidharar/Documents/School/thesis/git_repositories/')

# HUJI
sys.path.append('/sci/home/david.harar/Desktop/')
os.chdir('/sci/home/david.harar/Desktop/')



from combined_downstream_upstream.utils.LoadModels import *
from combined_downstream_upstream.executors.train_combined_model import trainer


# upstream model params and config
# ---------------
upstream_params = {
    'input_dimension': 12,
    'output_dimension': 12,              
    'hidden_dimmension':  128,           # d_model (int) – the number of expected features in the input (required)???,
    'attention_heads': 8,               # number of attention heads, if None then d_model//64,
    'encoder_number_of_layers': 8,
    'dropout': 0.4,
    'clip': 1,
    'positional_encodings': False,
    'device':'cuda'
}

# get global argumnents
# -----------------------------
parser = argparse.ArgumentParser(description="Run trainer with specified number of channels to turn off.")
parser.add_argument('--num_channels_to_turn_off', type=int, required=True, help='Number of channels to turn off')

args = parser.parse_args()


# run script
# -----------------------------
best_rocauc_and_pr_auc = {
    'channels_to_turn_off':[],
    'seed':[],
    'validation-roc-auc':[],
    'validation-pr-auc':[],
    'test-roc-auc':[],
    'test-pr-auc':[],
}

seed = 123
np.random.seed(123)
seeds = np.random.randint(0,1000, 15)

config = {
            # general
            'targets': ['AF'],
            'leads': ['LI', 'LII', 'LIII', 'aVL', 'aVR','aVF', 'V1','V2','V3','V4','V5','V6'],
            # 'leads': ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'],

            # training
            'batch_size': 128,
            'n_epochs': 10, 
            'weight_decay': 0.3,
            'lr': 0.0005,
            'eval_metric':'aucpr',
            'patience':3,
            'clip':1,
            'loss_function_weight':None,
            'predefined_device':'cuda',

            # Experiment settings
            'impute_only_missing':True,
            'continue_training_upstream_model':False,

            # test
            'check_on_test':True,
            'plot':False, # convert to true if want to plot
            }

for seed_ in seeds:

    print('-'*25)
    print(f'Turned off channels: {args.num_channels_to_turn_off}')
    print('-'*25)



    # Empty leads
    # -------------------
    config['seed']=seed_
    config['metadata_file_path']    = './downstream_classification/data/combined_data/china.csv'
    config['data_folder_path']      = './downstream_classification/data/china-processed_divided_into_450/'

    config['internal_data']         = False
    config['channels_to_turn_off']  = 0
    config['model_saving_path']     = f"./combined_downstream_upstream/models/china-{config['channels_to_turn_off']}_channels_off-no_additional_noise-impute_only_missing"
    config['plot_saving_path']      = f"/home/david/Desktop/projects/thesis/combined_downstream_upstream/plots/china-{config['channels_to_turn_off']}_channels_off-no_additional_noise-impute_only_missing"

    upstream_model   = load_upstream_model(upstream_params, folder_path = './upstream_seq2seq/models/', model_name = 'transformer_cnn_4heads', device = upstream_params['device'])
    downstream_model = load_downstream_model(
        dropout=0.5,scale=1,num_inputs=12, 
        weights_path=f"./downstream_classification/models/AF-V10-different-seeds-china-{config['channels_to_turn_off']}_channels_off-saving-downstream-models/model_val_aucpr_seed_17.pt",
        device = config['predefined_device']
        )

    config['upstream_model']=upstream_model
    config['downstream_model']=downstream_model


    # training:
    # --------------------
    best_rocauc_and_pr_auc_seed = trainer(**config)

    best_rocauc_and_pr_auc['channels_to_turn_off'].append(args.num_channels_to_turn_off)
    best_rocauc_and_pr_auc['seed'].append(config['seed'])
    best_rocauc_and_pr_auc['validation-roc-auc'].append(best_rocauc_and_pr_auc_seed['validation-roc-auc'])
    best_rocauc_and_pr_auc['validation-pr-auc'].append(best_rocauc_and_pr_auc_seed['validation-auc-pr'])
    best_rocauc_and_pr_auc['test-roc-auc'].append(best_rocauc_and_pr_auc_seed['test-roc-auc'])
    best_rocauc_and_pr_auc['test-pr-auc'].append(best_rocauc_and_pr_auc_seed['test-auc-pr'])

    with open(config['model_saving_path']+'/best_rocauc_and_pr_auc.pkl', 'wb') as f:
        pickle.dump(best_rocauc_and_pr_auc, f)




