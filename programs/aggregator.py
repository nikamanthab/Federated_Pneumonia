import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn


def weights_to_array(model):
    '''
        input: pytorch model
        output: array of tensors of model weights
    '''
    model_weights = []
    for param in model.parameters():
        model_weights.append(param) # check without .data
    return model_weights

def fed_avg_aggregator(model_data, args):
    '''
        input: array of tuples containing model 
        and the number of samples from each respective node
        output: fed_avg aggregated model
    '''
    total_no_samples = 0
    
    # creates an array of weights and sample counts
    # shape -> no_models*no_layers*dim_of_layer
    node_weights = []
    node_samples = []
    for model,no_samples in model_data:
        node_weights.append(weights_to_array(model))
        node_samples.append(no_samples)
    # calculates the total number of samples
        total_no_samples += no_samples
    
    aggregated_weights = []
    for layer_idx in range(len(node_weights[0])):
        temp = torch.zeros(node_weights[0][layer_idx].shape).to(args.device)
        for node_idx in range(len(node_weights)):
            temp+= (node_samples[node_idx]/total_no_samples)*node_weights[node_idx][layer_idx]
        aggregated_weights.append(temp)
    agg_model = args.model().to(args.device)
    for idx, param in enumerate(agg_model.parameters()):
        param.data = aggregated_weights[idx]
    return agg_model

# x = fed_avg_aggregator([(model1,60),(model2,40)])

#COMED aggregator
def comed_aggregator(model_data):
    '''
        input: array of tuples containing model 
        and the number of samples from each respective node
        output: fed_avg aggregated model
    '''
    total_no_samples = 0
    
    # creates an array of weights and sample counts
    # shape -> no_models*no_layers*dim_of_layer
    node_weights = []
    node_samples = []

    
    for model,no_samples in model_data:
        node_weights.append(weights_to_array(model))
        node_samples.append(no_samples)
        total_no_samples += no_samples
     
    aggregated_weights = []
    for layer_idx in range(len(node_weights[0])):
        layer_shape = node_weights[0][layer_idx].shape
        temp = torch.zeros(node_weights[0][layer_idx].shape)
        for node_idx in range(len(node_weights)):
            if(node_idx == 0):
                temp = torch.flatten(node_weights[node_idx][layer_idx]).unsqueeze(1)
            else:
                layer_flattened = torch.flatten(node_weights[node_idx][layer_idx]).unsqueeze(1)
                temp = torch.cat((temp, layer_flattened),1)
        temp = temp.detach().numpy()
        temp = np.median(temp,1)
        temp = torch.from_numpy(temp)
        temp = torch.reshape(temp, layer_shape)
        aggregated_weights.append(temp)

    agg_model = models.resnet18(pretrained=True)
    agg_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    agg_model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.Linear(in_features=256, out_features=128, bias=True),
                                nn.Linear(in_features=128, out_features=2, bias=True))
    for idx, param in enumerate(agg_model.parameters()):
        param.data = aggregated_weights[idx]
    return agg_model