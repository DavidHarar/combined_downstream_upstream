import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from downstream_classification.modeling.Inception import *
import torch
from tqdm import tqdm

def plot_roc_auc(y_pred, y_true,saving_path):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.title('ROC curve')
    plt.savefig(f'{saving_path}/roc.jpg')
    plt.clf()

def plot_prc(y_pred,y_true,saving_path):
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    plt.plot(recall, precision, label='ROC curve (area = %0.2f)' % auc_pr)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curve")
    plt.savefig(f'{saving_path}/prc.jpg')
    plt.clf()

def _get_sample(signal_index,data_path, data, targets=['one_year_until_death']):
    """
    load the relevant <lead> reading .npy file from <data_folder_path>. 
    Inputs:
    - signal_index: int
    - data_folder_path:str. The path to the folder when the desired file can be loaded from
    """

    file_name = f'{signal_index}.npy'
    X = np.load(data_path+file_name, allow_pickle=True)
    y = data[data['reading'] == signal_index][targets].values

    if len(y)>1:
        y = np.expand_dims(y[0],-1)
    return X,y

def preprocessing_a_reading(reading, data, targets=['one_year_until_death'], fillna=0):
    """
    All the preprocessing needed in order to be able to add a reading into the model    
    """
    x_test_reading, _ = _get_sample(reading,data_path='./data/individual-signals/', data=data, targets=targets)
    x_test_reading = np.float32(x_test_reading)
    x_test_reading = np.expand_dims(x_test_reading,0)
    x_test_reading = np.nan_to_num(x_test_reading, nan=fillna)
    x_test_reading = torch.from_numpy(x_test_reading)
    return x_test_reading


def predict(device, 
            readings, 
            model, 
            data, 
            targets, 
            fillna=0,
            leads = ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'], 
            data_path='./data/individual-signals/'):
    """
    Return scores for `redings` using `model`.    
    """
    leads_and_their_indices = {x:i for i,x in enumerate(['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'])}
    relevant_leads_indices = np.array([leads_and_their_indices[x] for x in leads])
    
    predictions = []
    nonmissing_leads = []
    model.eval()
    with torch.no_grad():

        # return [preprocessing_a_reading(reading, data, model=model, relevant_leads_indices=relevant_leads_indices, targets=['one_year_until_death'], fillna=0) for reading in readings]

        for reading in tqdm(readings, desc='Collecting scores'):
            # load reading
            x_test_reading, _ = _get_sample(
                reading,data_path=data_path, data=data, targets=targets
            )

            # encode
            x_test_reading = np.float32(x_test_reading)
                        
            # take only relevant leads
            x_test_reading = x_test_reading[relevant_leads_indices]

            # # adjust dimensions
            x_test_reading = np.expand_dims(x_test_reading,0) # batch, channels, length
            
            # keep the number non missing leads in reading
            num_nonmissing_leads = [np.count_nonzero(~np.isnan(x_test_reading[j][:,0])) for j in range(len(x_test_reading))]

            # fix shapes (convert into -> [length, batch_size, channels])
            x_test_reading = np.float32(np.transpose(x_test_reading, axes=(2,0,1)))

            # fillna
            x_test_reading = np.nan_to_num(x_test_reading, nan=fillna)

            # encode as torch tensor
            x_test_reading = torch.from_numpy(x_test_reading)
            
            # gpu
            x_test_reading = x_test_reading.to(device)

            predictions.append(model(x_test_reading).item())
            nonmissing_leads.append(num_nonmissing_leads)

    nonmissing_leads = np.array(nonmissing_leads).reshape(-1)
    return predictions, nonmissing_leads


def post_reg_analysis(data, y_true_column, y_pred_column, saving_path):

    # plot results
    # ----------------
    
    # Scores on Validation
    sns.histplot(data = data, x=y_pred_column, hue = y_true_column, common_norm=False, stat='probability')
    plt.title(f'Scores Distribution wrt wrt label')
    plt.savefig(f'{saving_path}/scores_on_validation.jpg')
    plt.clf()

    sns.histplot(data = data, x=y_pred_column, hue = 'one_year_until_death', common_norm=False, stat='probability')
    plt.title(f'Scores Distribution wrt wrt one_year_until_death')
    plt.savefig(f'{saving_path}/scores_on_validation_wrt_one_year_until_death.jpg')
    plt.clf()

    # ROC
    plot_roc_auc(data[y_pred_column], data[y_true_column], saving_path)
    
    # PR-Curve
    plot_prc(data[y_pred_column], data[y_true_column], saving_path)

    # preprocessing for scores plots    
    data['age_in_years'] = np.round(data['age_days']/365)
    data['age_window'] = pd.cut(data['age_in_years'], bins = [0,50,55,60,65,70,75,80,85,90,95,110,np.inf])
    data['time_until_death_window'] = pd.cut(data['days_until_approximate_death'], bins=[0,20,100,250,500,1000,2000,3000,4000,np.inf])
    
    # Average Scores per age window
    sns.boxplot(data = data, y = y_pred_column, x = 'age_window',showfliers = False, color='grey')
    plt.xticks(rotation = 90)
    plt.savefig(f'{saving_path}/avg_score_per_age_window.jpg')
    plt.clf()

    # Scores averages per time-until-death
    sns.boxplot(data = data, y = y_pred_column, x = 'time_until_death_window',showfliers = False, color='grey')
    plt.xticks(rotation = 90)
    plt.savefig(f'{saving_path}/scores_boxes_time_until_death_window.jpg')
    plt.clf()

    # By target
    sns.boxplot(data = data, y = y_pred_column, x = y_true_column, showfliers = False, color='grey')
    plt.xticks(rotation = 90)
    plt.savefig(f'{saving_path}/scores_boxes_by_target.jpg')
    plt.clf()

    # Scores per age wrt target
    sns.boxplot(data = data, y = y_pred_column, x = 'age_window', hue = y_true_column, showfliers = False, color='grey')
    plt.xticks(rotation = 90)
    plt.savefig(f'{saving_path}/scores_boxes_by_age_window_hued_target.jpg')
    plt.clf()
