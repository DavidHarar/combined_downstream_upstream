import os
import torch
from upstream_seq2seq.modeling.Transformer import TSTransformerEncoderCNN

def get_best_upsteam_model(folder_path, model_name):
    """
    Since that in the upstream stage we save basically every model, it is important to 
    take the one that minimized the loss over the validation set.
    """
    model_files = [x for x in os.listdir(folder_path) if model_name in x and 'loss' not in x]

    # get timestemps
    timestemps = [float(x.replace(f'{model_name}_', '').replace('.pt','')) for x in model_files]
    timestemps.sort()

    # get best model (the last one that has been saved)
    best_model_path = folder_path+model_name+'_'+str(timestemps[-1])+'.pt'
    return best_model_path


def load_upstream_model(params, folder_path, model_name):
    upstream_path = get_best_upsteam_model(folder_path, model_name)
    upstream_model = TSTransformerEncoderCNN(
        params['input_dimension'], 
        params['output_dimension'], 
        params['hidden_dimmension'], 
        params['attention_heads'], 
        params['encoder_number_of_layers'], 
        params['dropout'])
    upstream_model.load_state_dict(torch.load(upstream_path))
    return upstream_model