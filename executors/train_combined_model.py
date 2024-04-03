import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from downstream_classification.dataloader.DataGenerator import DataGenerator
from combined_downstream_upstream.modeling.JointModel import CombinedModel
# from downstream_classification.dataloader.DataLoader import DataGenerator
# from combined_downstream_upstram.modeling.JointModel import CombinedModel

from combined_downstream_upstream.utils.plots import plot_test_signals_12leads_SHL


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from tqdm.auto import tqdm
from downstream_classification.utils.metrics import *
from combined_downstream_upstream.utils.plots import *
from sklearn.model_selection import ParameterGrid

import os

import logging


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
            targets,
            batch_size, 
            n_epochs,
            clip,

            impute_only_missing,
            upstream_model,
            downstream_model,
            continue_training_upstream_model,
            model_saving_path,

            leads = ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'],
            eval_metric = 'loss',
            weight_decay = 0,
            lr = 0.001,
            patience=np.inf,
            loss_function_weight = None,

            check_on_test = False,

            # plot
            plot=False,
            plot_saving_path=None,
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

    
    # extract target string
    target_str = targets[0] if isinstance(targets,list) else targets

    # Init experiment directory
    if not os.path.exists(model_saving_path):
        os.makedirs(model_saving_path)

    # init log
    logging.basicConfig(filename=f"./{model_saving_path}/log.log", level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M')
    logging.info("Fit the preprocessing pipeline")

    

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
    logging.info(f'Training using device: {device}')

    # convert loss_function_weight
    if loss_function_weight:
       loss_function_weight = torch.tensor([loss_function_weight]).to(device)


    # Create Generators
    # ------------------
    logging.info(f'Creating generators')
    train_generator = DataGenerator(
        metadata_file_path= metadata_file_path,                 # path to metadata file
        data_folder_path = data_folder_path,                    # path to individual signals
        sample='train',                                         # sample we want to create a generator to. Either train, validation or test
        targets=targets,                                        # list of targets we want train on
        batch_size=batch_size,                                  # batch size
        shuffle=True,                                            # Whether to shuffle the list of IDs at the end of each epoch.
        seed = seed
                    )
    validation_generator = DataGenerator(
        metadata_file_path= metadata_file_path,                 # path to metadata file
        data_folder_path = data_folder_path,                    # path to individual signals
        sample='validation',                                    # sample we want to create a generator to. Either train, validation or test
        targets=targets,                                        # list of targets we want train on
        batch_size=batch_size,                                  # batch size
        shuffle=True,                                            # Whether to shuffle the list of IDs at the end of each epoch.
        seed = seed
                    )

    test_generator = DataGenerator(
        metadata_file_path= metadata_file_path,                 # path to metadata file
        data_folder_path = data_folder_path,                    # path to individual signals
        sample='test',                                          # sample we want to create a generator to. Either train, validation or test
        targets=targets,                                        # list of targets we want train on
        batch_size=batch_size,                                  # batch size
        shuffle=False,                                            # Whether to shuffle the list of IDs at the end of each epoch.
        seed = seed
                                )
    
    
    
    # create a model
    # ------------------
    model = CombinedModel(upstream_model, downstream_model, device, continue_training_upstream_model, impute_only_missing)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=lr)
    print(count_parameters(model))


    if len(targets)>1:
        criterion = nn.CrossEntropyLoss()
    else:
        if loss_function_weight:
            criterion = nn.BCELoss(loss_function_weight)
        else: 
            criterion = nn.BCELoss()

    logging.info(f'The model has {count_parameters(model):,} trainable parameters')
    logging.info('* Model:')
    logging.info('* -----------')
    logging.info(model)
    logging.info('* -----------')


    # Training
    # ------------------
    
    # initiate values
    best_valid_loss = float('inf')
    best_aucpr = 0
    best_rocauc = 0
    
    rocauc_given_best_aucpr = 0
    epochs_without_update = 0   
    best_recall_for_precision = 0

    losses = {'train':[],
              'validation':[]}

    
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
        
        # test
        if check_on_test:
            test_loss, y_test, y_test_pred = evaluate_epoch(model, 
                                test_generator, 
                                criterion, 
                                device
                                )
            
            if plot:
                plot_test_signals_12leads_SHL(model.upstream_model, 
                                    test_generator, 
                                    device, 
                                    epoch,
                                    plot_saving_path=f'{plot_saving_path}')
                
        # store losses
        losses['train'].append(train_loss)
        losses['validation'].append(valid_loss)


        # Plot distributions
        y_train_prediction = pd.DataFrame({'y_train': y_train,
                                           'y_train_pred':y_train_pred})
        
        y_valication_prediction = pd.DataFrame({'y_val': y_val,
                                               'y_val_pred':y_val_pred})
        
        if check_on_test:
            y_test_prediction = pd.DataFrame({'y_test': y_test,
                                                'y_test_pred':y_test_pred})
                
        # save loss
        if model_saving_path:
            with open(f'{model_saving_path}/loss.pkl', 'wb') as f:
                pickle.dump(losses, f)
        
        # take ending time
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        aucpr  = PRAUC(y_val, y_val_pred)
        rocauc = ROCAUC(y_val, y_val_pred)
        recall_for_precision, threshold = MaxRecall_for_MinPrecision(y_val, y_val_pred, min_precision=0.4)

        # results on test
        if check_on_test:
            aucpr_test  = PRAUC(y_test, y_test_pred)
            rocauc_test = ROCAUC(y_test, y_test_pred)
        
        # patience
        if eval_metric == 'loss':
            if valid_loss < best_valid_loss:
                epochs_without_update = 0
            else:
                epochs_without_update+=1
        if eval_metric == 'aucpr':
            if aucpr > best_aucpr:
                epochs_without_update = 0
            else:
                epochs_without_update+=1
        if eval_metric == 'rocauc':
            if rocauc > best_rocauc:
                epochs_without_update = 0
            else:
                epochs_without_update+=1
        if eval_metric == 'recall_for_precision':
            if recall_for_precision > best_recall_for_precision:
                epochs_without_update = 0
            else:
                epochs_without_update+=1

        # break if patience condition takes place
        if epochs_without_update>patience:
            break
        

        
        # terminal plots saving
        if eval_metric == 'loss':
            if valid_loss < best_valid_loss:
                if model_saving_path is not None:
                    fig, axs = plt.subplots(1, 2, figsize = (10,3))
                    sns.histplot(data = y_train_prediction, x = 'y_train_pred', hue = 'y_train', common_norm=False, stat='probability', ax=axs[0])
                    sns.histplot(data = y_valication_prediction, x = 'y_val_pred', hue = 'y_val', common_norm=False, stat='probability', ax=axs[1])
                    axs[0].set_title('Scores Distribution on the Training Set')
                    axs[1].set_title('Scores Distribution on the Validation Set')
                    axs[0].axvline(threshold, c='r')
                    axs[1].axvline(threshold, c='r')
                    fig.savefig(f'{model_saving_path}/epoch_{epoch}_val_loss.jpg')
                    plt.cla()
                    # plt.show()

        if eval_metric == 'aucpr':
            if aucpr > best_aucpr:
                if model_saving_path is not None:
                    fig, axs = plt.subplots(1, 2, figsize = (10,3))
                    sns.histplot(data = y_train_prediction, x = 'y_train_pred', hue = 'y_train', common_norm=False, stat='probability', ax=axs[0])
                    sns.histplot(data = y_valication_prediction, x = 'y_val_pred', hue = 'y_val', common_norm=False, stat='probability', ax=axs[1])
                    axs[0].set_title('Scores Distribution on the Training Set')
                    axs[1].set_title('Scores Distribution on the Validation Set')
                    axs[0].axvline(threshold, c='r')
                    axs[1].axvline(threshold, c='r')
                    fig.savefig(f'{model_saving_path}/epoch_{epoch}_val_aucpr.jpg')
                    plt.cla()
                    # plt.show()

        if eval_metric == 'rocauc':
            if rocauc > best_rocauc:
                if model_saving_path is not None:
                    fig, axs = plt.subplots(1, 2, figsize = (10,3))
                    sns.histplot(data = y_train_prediction, x = 'y_train_pred', hue = 'y_train', common_norm=False, stat='probability', ax=axs[0])
                    sns.histplot(data = y_valication_prediction, x = 'y_val_pred', hue = 'y_val', common_norm=False, stat='probability', ax=axs[1])
                    axs[0].set_title('Scores Distribution on the Training Set')
                    axs[1].set_title('Scores Distribution on the Validation Set')
                    axs[0].axvline(threshold, c='r')
                    axs[1].axvline(threshold, c='r')
                    fig.savefig(f'{model_saving_path}/epoch_{epoch}_val_rocauc.jpg')
                    plt.cla()
                    # plt.show()

        if eval_metric == 'recall_for_precision':
            if recall_for_precision > best_recall_for_precision:
                if model_saving_path is not None:
                    fig, axs = plt.subplots(1, 2, figsize = (10,3))
                    sns.histplot(data = y_train_prediction, x = 'y_train_pred', hue = 'y_train', common_norm=False, stat='probability', ax=axs[0])
                    sns.histplot(data = y_valication_prediction, x = 'y_val_pred', hue = 'y_val', common_norm=False, stat='probability', ax=axs[1])
                    axs[0].set_title('Scores Distribution on the Training Set')
                    axs[1].set_title('Scores Distribution on the Validation Set')
                    axs[0].axvline(threshold, c='r')
                    axs[1].axvline(threshold, c='r')
                    fig.savefig(f'{model_saving_path}/epoch_{epoch}_val_recall_for_precision.jpg')
                    plt.cla()


        # update best values
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
            if check_on_test:
                best_test_loss = test_loss
            
            # save if val_loss is criterion for saving
            if eval_metric == 'loss':
                if model_saving_path:
                    torch.save(model.state_dict(), f'{model_saving_path}/model_val_loss.pt')

        if aucpr > best_aucpr:
            best_aucpr = aucpr
            
            if check_on_test:
                best_test_aucpr = aucpr_test
            
            # save if val_loss is criterion for saving
            if eval_metric == 'aucpr':
                if model_saving_path:
                    torch.save(model.state_dict(), f'{model_saving_path}/model_val_aucpr.pt')

        if rocauc > best_rocauc:
            best_rocauc = rocauc

            if check_on_test:
                best_test_rocauc = rocauc_test

            # save if val_loss is criterion for saving
            if eval_metric == 'rocauc':
                if model_saving_path:
                    torch.save(model.state_dict(), f'{model_saving_path}/model_val_rocauc.pt')

        if recall_for_precision > best_recall_for_precision:
            best_recall_for_precision = recall_for_precision
            # save if val_loss is criterion for saving
            if eval_metric == 'recall_for_precision':
                if model_saving_path:
                    torch.save(model.state_dict(), f'{model_saving_path}/model_val_recall_for_precision.pt')

        if eval_metric == 'recall_for_precision':
            if recall_for_precision == best_recall_for_precision:
                best_value = best_recall_for_precision
                update_about_it = True
        if eval_metric == 'aucpr':
            if aucpr == best_aucpr:
                best_value = best_aucpr
                update_about_it = True
        if eval_metric == 'rocauc':
            if rocauc == best_rocauc:
                best_value = best_rocauc
                update_about_it = True
        if eval_metric == 'loss':
            if valid_loss == best_valid_loss:
                best_value = best_valid_loss
                update_about_it = True
        
        # Summarize epoch results
        logging.info('-'*45)
        logging.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        if update_about_it:
            logging.info(f'\t New best val_rocauc loss was found, current best value is {np.round(best_value,5)}')
            update_about_it = False
        logging.info(f'\t Train Loss: {train_loss:.3f}')
        logging.info(f'\t Val. Loss: {valid_loss:.3f}')
        logging.info(f'\t ROC-AUC: {rocauc:.3f}')
        logging.info(f'\t PR-AUC: {aucpr:.3f}')
        logging.info(f'\t Best Val. Loss: {best_valid_loss:.3f}')
        logging.info(f'\t Best ROC-AUC: {best_rocauc:.3f}')
        logging.info(f'\t Best PR-AUC: {best_aucpr:.3f}')
        if check_on_test:
            logging.info(f'\t Test-ROC-AUC under Best Validation ROC-AUC: {best_test_rocauc:.3f}')
            logging.info(f'\t Test-PR-AUC under Best Validation Best PR-AUC: {best_test_aucpr:.3f}')
        
        logging.info('-'*45)

    # get best model
    best_model = CombinedModel(upstream_model, downstream_model, device, continue_training_upstream_model, impute_only_missing)
    best_model = best_model.to(device)
    if eval_metric == 'recall_for_precision':
        best_model.load_state_dict(torch.load(f'{model_saving_path}/model_val_recall_for_precision.pt'))
    if eval_metric == 'aucpr':
        best_model.load_state_dict(torch.load(f'{model_saving_path}/model_val_aucpr.pt'))
    if eval_metric == 'rocauc':
        best_model.load_state_dict(torch.load(f'{model_saving_path}/model_val_rocauc.pt'))
    if eval_metric == 'loss':
        best_model.load_state_dict(torch.load(f'{model_saving_path}/model_val_loss.pt'))
    

    # save additional plots
    validation_data = pd.read_csv(metadata_file_path,index_col=0)
    validation_data = validation_data[validation_data['sample'] == 'validation'].reset_index(drop=True)

    # best_model = best_model.to('cpu')
    predictions, nonmissing_leads = predict(
        device = device,
        readings= validation_data['reading'],
        model = best_model,
        data = validation_data,
        targets=targets,
        fillna=0,
        leads = ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'],
        data_path=data_folder_path,
    )


    validation_data['y_pred'] = predictions
    post_reg_analysis(
        data = validation_data,
        y_true_column=target_str,
        y_pred_column='y_pred',
        saving_path=model_saving_path
    )

    logging.shutdown()

    if check_on_test:
        return {'validation-roc-auc': best_rocauc, 'validation-auc-pr': best_aucpr, 'test-roc-auc': best_test_rocauc, 'test-auc-pr': best_test_aucpr}

    else:
        return {'roc-auc': best_rocauc, 'auc-pr': best_aucpr}




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
            'leads': ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'],
            'eval_metric': 'aucpr',
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

        return results



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
        X, y, _ = next(it)
        # y=np.squeeze(y,-1)

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
            X, y, _ = next(it)
            # y=np.squeeze(y,-1)
            
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