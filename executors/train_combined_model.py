import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from downstream_classification.dataloader.DataLoader import DataGenerator
from combined_downstream_upstram.modeling.JointModel import CombinedModel


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from tqdm.auto import tqdm
from downstream_classification.utils.metrics import *
from sklearn.model_selection import ParameterGrid

import os

# to be moved to utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def process(X):
    src_ = np.float32(np.transpose(X, axes=(2,0,1)))
    src_ = torch.from_numpy(src_).to(device)
    return src_
# ----

def trainer(seed,                    # seed
            metadata_file_path,      # metadata file include 
            data_folder_path, 
            fillna,
            targets,
            batch_size, 
            n_epochs,
            saving_path,
            clip,
            impute_only_missing,
            upstream_model,
            downstream_model,
            continue_training_upstream_model,
            minimal_number_of_leads=None,
            leads = ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'],
            eval_metric = 'loss',
            weight_decay = 0,
            lr = 0.001,
            verbosity = False,
            patience=np.inf,
            loss_function_weight = None,
         ):
    """
    Train an experiment, save results optionally.


    Inputs:
    - impute_only_missing: when in the downstream model, for the empty leads we should use the imputation. For the non empty leads we can either
        use the imputed or the original ones. If <impute_only_missing> is True, the data that will be entered to the downstream model is the origianl
        signals when possible, and imputed signals when the original signal was zero. If False, the downstream data will use all the leads as they were
        imputed. 
    - upstream_model: A pre-trained upstream model. Can be either trained together with the downstream or not.
    - downstream_model: A downstream model to be trained.
    """
    
    # TODO:
    # -----------------
    # 1. Fix saving path. We want to save best plot, save best model, save history, in a folder assiciated with params
    # 2. Return results and params so we can append it with the rest.



    # create a mapping for all possible leads
    # ------------------
    leads_and_their_indices = {x:i for i,x in enumerate(['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'])}
    relevant_leads_indices = np.array([leads_and_their_indices[x] for x in leads])

    # Fix randomness
    # ------------------
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    print('\n')
    print(f'training using device: {device}')
    print('\n')

    # convert loss_function_weight
    if loss_function_weight:
       loss_function_weight = torch.tensor([loss_function_weight]).to(device)


    # Create Generators
    # ------------------
    train_generator = DataGenerator(
        metadata_file_path= metadata_file_path,                 # path to metadata file
        data_folder_path = data_folder_path,                    # path to individual signals
        sample='train',                                         # sample we want to create a generator to. Either train, validation or test
        targets=targets,                                        # list of targets we want train on
        batch_size=batch_size,                                  # batch size
        shuffle=True,                                            # Whether to shuffle the list of IDs at the end of each epoch.
        fillna = fillna,
        minimal_number_of_leads = minimal_number_of_leads,
        leads = relevant_leads_indices
                    )


    validation_generator = DataGenerator(
        metadata_file_path= metadata_file_path,                 # path to metadata file
        data_folder_path = data_folder_path,                    # path to individual signals
        sample='validation',                                    # sample we want to create a generator to. Either train, validation or test
        targets=targets,                                        # list of targets we want train on
        batch_size=batch_size,                                  # batch size
        shuffle=True,                                            # Whether to shuffle the list of IDs at the end of each epoch.
        fillna = fillna,
        minimal_number_of_leads=minimal_number_of_leads,
        leads = relevant_leads_indices
                    )

    test_generator = DataGenerator(
        metadata_file_path= metadata_file_path,                 # path to metadata file
        data_folder_path = data_folder_path,                    # path to individual signals
        sample='test',                                          # sample we want to create a generator to. Either train, validation or test
        targets=targets,                                        # list of targets we want train on
        batch_size=batch_size,                                  # batch size
        shuffle=True,                                            # Whether to shuffle the list of IDs at the end of each epoch.
        fillna = fillna,
        minimal_number_of_leads=minimal_number_of_leads,
        leads = relevant_leads_indices
                                )
    
    
    
    # create a model
    # ------------------
    


    model = CombinedModel(impute_only_missing)
    print(count_parameters(model))
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=lr)


    if len(targets)>1:
        criterion = nn.CrossEntropyLoss()
    else:
        if loss_function_weight:
            criterion = nn.BCELoss(loss_function_weight)
        else: 
            criterion = nn.BCELoss()

    if verbosity:
        print(f'The model has {count_parameters(model):,} trainable parameters')
        print(model)


    # Training
    # ------------------
    
    # Initiate values
    best_valid_loss = float('inf')
    best_aucper = 0
    losses = {'train':[],
              'validation':[]}
    epochs_without_update = 0
    # best_aucper, rocauc_given_best_aucpr, tprforbudget_given_best_aucpr = None, None, None
    
    # Training loop
    for epoch in range(n_epochs):
        
        # take starting time
        start_time = time.time()
        
        # train
        # number of leads (dictates training scheme):
        train_loss, y_train, y_train_pred = train_epoch(model, 
                           train_generator, 
                           optimizer, 
                           criterion, 
                           clip, 
                           device
                           )
        

        # evaluate
        valid_loss, y_val, y_val_pred = evaluate_epoch(model, 
                              validation_generator, 
                              criterion, 
                              device
                              )
        
        # store losses
        losses['train'].append(train_loss)
        losses['validation'].append(valid_loss)


        # Handle Resutls
        
        ## Get KPIs
        aucpr  = PRAUC(y_val, y_val_pred)
        aucroc = ROCAUC(y_val, y_val_pred)
        tprforbudget = get_tpr_for_fpr_budget(y_val, y_val_pred, fpr_budget = 0.6)
        
        ## Save loss
        try:
            os.mkdir(saving_path)
        except:
            pass
        with open(f'{saving_path}/loss.pkl', 'wb') as f: # loss is saved regardless of results
            pickle.dump(losses, f)
        
        ## keep best model
        if eval_metric == 'loss':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                if saving_path:
                    print(f'New best validation loss was found, current best valid loss is {np.round(best_valid_loss,4)}')
                    torch.save(model.state_dict(), f'{saving_path}/model.pt')

                    # save plots
                    plot_maker(y_train, y_train_pred, y_val, y_val_pred, saving_path, '')

                epochs_without_update = 0 # if we update then patience restarts
            else:
                epochs_without_update+=1
                

        if eval_metric == 'aucpr':
            if aucpr>best_aucper:
                try:
                    best_aucper = aucpr
                    rocauc_given_best_aucpr = aucroc
                    tprforbudget_given_best_aucpr = tprforbudget
                except:
                    best_aucper = aucpr
                
                if saving_path:
                    # save model
                    print(f'New best aucpr was found, current best aucpr is {np.round(best_aucper,4)}')
                    torch.save(model.state_dict(), f'{saving_path}/model.pt')

                    # save plots
                    plot_maker(y_train, y_train_pred, y_val, y_val_pred, saving_path, '')

                epochs_without_update = 0 # if we update then patience restarts

            else:
                epochs_without_update+=1
        
        if epochs_without_update>patience:
            break
        

        # take ending time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        
        # print summary
        print('-'*45)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        print(f'\t ROC-AUC: {aucroc:.3f}')
        print(f'\t PR-AUC: {aucpr:.3f}')
        print(f'\t TPR for FPR=0.6 Budget: {tprforbudget:.3f}')
        print(f'\t Best Val. Loss: {best_valid_loss:.3f}')
        print('-'*45)

    
    return best_aucper, rocauc_given_best_aucpr, tprforbudget_given_best_aucpr




# TODO: move to utils
def plot_maker(y_train, y_train_pred, y_val, y_val_pred, path, title):
    """
    save discrimination for model
    """
    # Plot distributions
    y_train_prediction = pd.DataFrame({'y_train': y_train,
                                        'y_train_pred':y_train_pred})
    
    y_valication_prediction = pd.DataFrame({'y_val': y_val,
                                            'y_val_pred':y_val_pred})

    fig, axs = plt.subplots(1, 2, figsize = (10,3))
    sns.histplot(data = y_train_prediction, x = 'y_train_pred', hue = 'y_train', common_norm=False, stat='probability', ax=axs[0])
    sns.histplot(data = y_valication_prediction, x = 'y_val_pred', hue = 'y_val', common_norm=False, stat='probability', ax=axs[1])
    axs[0].set_title('Scores Distribution on the Training Set')
    axs[1].set_title('Scores Distribution on the Validation Set')
    fig.suptitle(title)
    plt.savefig(f'{path}/PDF.png')


def run_experiments(other_vars, constants, variables):
    """
    run multiple experiments and save results. 
    - other_vars - variables that are not suposed to be passed to trainer
    - constant - the constant parameters
    - variables - variable parameters

    # Example:
    # -------------

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
            'targets': ['one_year_until_death','DM','AF']
        }

        other_vars = {
            'notebook': '0.4',
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }

        run_experiments(other_vars, constants, variables)

    """
    # params = {**constants, **variables}
    # params[]
    grid = ParameterGrid(variables)

    results = {
        'short_description':[],
        'long_description':[],
        'aucper':[], 
        'rocauc':[], 
        'tprforbudget_given_best_aucpr':[]
    }

    for var_params in grid:
        folder_name = '-'.join([f'{k}-{var_params[k]}' for k in var_params.keys()])
        print(var_params)
        saving_path = f'./combined_downstream_upstram/models/comined_model_training/notebook-{other_vars["notebook"]}/{folder_name}'
        if not folder_name in os.listdir(f'./combined_downstream_upstram/models/comined_model_training/notebook-{other_vars["notebook"]}/'):
            os.makedirs(saving_path)
            

        params = {**constants, **var_params}
        params['saving_path'] = saving_path
        
        print('*'*len(f'* {folder_name} *'))
        print(f'* {folder_name} *')
        print('*'*len(f'* {folder_name} *'))
        
        long_description = '-'.join([f'{k}-{params[k]}' for k in params.keys()])

        best_aucper, rocauc_given_best_aucpr, tprforbudget_given_best_aucpr = trainer(**params)
        
        results['short_description'].append(folder_name)
        results['long_description'].append(long_description)
        results['aucper'].append(best_aucper)
        results['rocauc'].append(rocauc_given_best_aucpr)
        results['tprforbudget_given_best_aucpr'].append(tprforbudget_given_best_aucpr)
        
        pd.DataFrame(results).to_csv(f'./combined_downstream_upstram/models/comined_model_training/notebook-{other_vars["notebook"]}/summary.csv')
        
        print('')



def train_epoch(
        model, 
        iterator, 
        optimizer, 
        criterion, 
        clip, 
        device, 
          ):
    
    # set model on training state and init epoch loss    
    model.train()
    epoch_loss = 0
    denominator = 0
    ys = []
    outputs = []

    # get number of iterations for the progress bar. 
    it = iter(iterator)
    T = len(iterator)
    # set progress bar
    t = trange(T, desc='Within epoch loss (training)', leave=True)

    for i in t:
        # get data
        X, y = next(it)
        y=np.squeeze(y,-1)

        # don't run if there are NaNs
        if np.isnan(X).sum()>0:
            print('skipping because of NaNs')
            continue
        y = np.float32(y)

        X = process(X)    
        # X = torch.from_numpy(X)
        y = torch.from_numpy(y).type(torch.FloatTensor)

        X = X.to(device)
        y = y.to(device)
        
        # step
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        j = np.round(epoch_loss/(i+1),5)
        t.set_description(f"Within epoch loss (training) {j}")
        t.refresh() # to show immediately the update

        # keep y and output
        ys += y.to('cpu').numpy().reshape(-1).tolist()
        outputs += output.detach().to('cpu').numpy().reshape(-1).tolist()
            

        denominator+=1

    return epoch_loss / denominator, np.array(ys), np.array(outputs)

def evaluate_epoch(model, 
                   iterator, 
                   criterion,
                   device):
    
    # set model on training state and init epoch loss    
    model.eval()
    epoch_loss = 0
    denominator = 0

    # get number of iterations for the progress bar. n_iters can be set to bypass it
    it = iter(iterator)
    T = len(iterator)
    # set progress bar
    t = trange(T, desc='Within epoch loss (validation)', leave=True)
    
    
    ys = []
    outputs = []

    with torch.no_grad():
        for i in t:

            # get data
            X, y = next(it)
            y=np.squeeze(y,-1)
            
            # don't run if there are NaNs
            if np.isnan(X).sum()>0:
                print('skipping because of NaNs')
                continue
            y = np.float32(y)

            X = process(X)    
            # X = torch.from_numpy(X)
            y = torch.from_numpy(y)

            X = X.to(device)
            y = y.to(device)
            
            output = model(X)
            loss = criterion(output, y)
            epoch_loss += loss.item()

            j = np.round(epoch_loss/(i+1),5)
            t.set_description(f"Within epoch loss (validation) {j}")
            t.refresh() # to show immediately the update

            # update values
            denominator+=1


            # keep y and output
            ys += y.to('cpu').numpy().reshape(-1).tolist()
            outputs += output.to('cpu').numpy().reshape(-1).tolist()
            
    return epoch_loss / denominator, np.array(ys), np.array(outputs)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)