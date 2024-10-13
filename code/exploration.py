import os
import sys
sys.path.append('/home/david/Desktop/projects/thesis/')
# os.chdir('/home/david/Desktop/projects/thesis/')

import numpy as np
import pickle 

from combined_downstream_upstream.utils.LoadModels import *
from combined_downstream_upstream.executors.train_combined_model import trainer
N_EPOCHS = 7



# upstream model params
# ---------------
upstream_params = {
    'input_dimension': 12,
    'output_dimension': 12,              
    'hidden_dimmension':  128,           # d_model (int) – the number of expected features in the input (required)???,
    'attention_heads': 8,                # number of attention heads, if None then d_model//64,
    'encoder_number_of_layers': 8,
    'dropout': 0.4,
    'clip': 1,
    'positional_encodings': False,
    'device':'cuda'
}
best_rocauc_and_pr_auc = {
    'seed':[],
    'validation-roc-auc':[],
    'validation-pr-auc':[],
    'test-roc-auc':[],
    'test-pr-auc':[],
}

seed = 123
np.random.seed(123)
seeds = np.random.randint(0,1000, 50)


seed = seeds[0]

print('-'*25)
print('Seed:', seed)
print('-'*25)

upstream_model   = load_upstream_model(upstream_params, folder_path = './upstream_seq2seq/models/', model_name = 'transformer_cnn_4heads')
downstream_model = load_downstream_model(dropout=0.5,scale=1,num_inputs=12, weights_path='./downstream_classification/models/AF-V9/model_val_rocauc.pt')


for reconstruction_loss in [0.2, 0.5,0.7]:
    for lr in [0.05,0.005,0.001]:
        for weight_decay in [0,0.3]:
            for loss_function_weight in [3,5,10]:
                for impute_only_missing in [True, False]:
                    comb = f'{reconstruction_loss}-{lr}-{weight_decay}-{loss_function_weight}-{impute_only_missing}'
                    config = {
                            # general
                            'seed':seed,
                            'metadata_file_path': './downstream_classification/data/combined_data/metadata_only_existant_readings_09042023.csv',
                            'data_folder_path': './downstream_classification/data/individual-signals-registered/',
                            'targets': ['AF'],
                            'leads': ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'],

                            # training
                            'batch_size': 128,
                            'n_epochs': N_EPOCHS,
                            'weight_decay': weight_decay, #0.3,
                            'lr': lr,
                            'eval_metric':'aucpr',
                            'patience':2,
                            'clip':1,
                            'loss_function_weight':loss_function_weight,

                            # Experiment settings
                            'upstream_model':upstream_model,
                            'downstream_model':downstream_model,
                            'impute_only_missing':impute_only_missing,
                            'continue_training_upstream_model':True,
                            'model_saving_path':f'./combined_downstream_upstream/models/other-experiments/{comb}',

                            'reconstruction_loss_weight':[reconstruction_loss]*N_EPOCHS,

                            # test
                            'check_on_test':True,
                            'plot':False,
                            'plot_saving_path':f'/home/david/Desktop/projects/thesis/combined_downstream_upstream/plots/other-experiments/{comb}',
                            # 'training_steps': 100
                            'training_steps': np.inf
                            }

                    best_rocauc_and_pr_auc_seed = trainer(**config)

                    best_rocauc_and_pr_auc['seed'].append(seed)
                    best_rocauc_and_pr_auc['validation-roc-auc'].append(best_rocauc_and_pr_auc_seed['validation-roc-auc'])
                    best_rocauc_and_pr_auc['validation-pr-auc'].append(best_rocauc_and_pr_auc_seed['validation-auc-pr'])
                    best_rocauc_and_pr_auc['test-roc-auc'].append(best_rocauc_and_pr_auc_seed['test-roc-auc'])
                    best_rocauc_and_pr_auc['test-pr-auc'].append(best_rocauc_and_pr_auc_seed['test-auc-pr'])

                    with open(config['model_saving_path']+'/best_rocauc_and_pr_auc.pkl', 'wb') as f:
                        pickle.dump(best_rocauc_and_pr_auc, f)
