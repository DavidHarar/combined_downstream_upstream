import os
import torch
from upstream_seq2seq.modeling.Transformer import TSTransformerEncoderCNN
from downstream_classification.modeling.Inception import *

def get_best_upsteam_model(folder_path = './upstream_seq2seq/models/', model_name = 'transformer_cnn_4heads'):
    """
    Since that in the upstream stage we save basically every model, it is important to 
    take the one that minimized the loss over the validation set.
    """
    model_files = [x for x in os.listdir(folder_path) if model_name in x and 'loss' not in x]

    # get timestemps
    timestemps = [float(x.replace('transformer_cnn_4heads_', '').replace('.pt','')) for x in model_files if x.endswith('.pt')]
    timestemps.sort()

    # get best model (the last one that has been saved)
    best_model_path = folder_path+model_name+'_'+str(timestemps[-1])+'.pt'
    return best_model_path


def load_upstream_model(params, folder_path, model_name, upstream_path = None):
    upstream_path = get_best_upsteam_model(folder_path, model_name)
    print('Upstream Model File Path:', upstream_path)
    upstream_model = TSTransformerEncoderCNN(
        params['input_dimension'], 
        params['output_dimension'], 
        params['hidden_dimmension'], 
        params['attention_heads'], 
        params['encoder_number_of_layers'], 
        params['dropout'])
    upstream_model.load_state_dict(torch.load(upstream_path))
    return upstream_model

def load_downstream_model(dropout, scale, num_inputs, weights_path):
    model = DownstreamInception(dropout=dropout, scale=scale, num_inputs=num_inputs).to('cuda')
    model.load_state_dict(torch.load(weights_path))
    return model
