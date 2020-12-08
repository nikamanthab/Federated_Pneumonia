import torch
from models.CNN_basic import TwoLayerNet

model1 = TwoLayerNet()
model2 = TwoLayerNet()

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
    agg_model = TwoLayerNet()
    for idx, param in enumerate(agg_model.parameters()):
        param.data = aggregated_weights[idx]
    return agg_model

# x = fed_avg_aggregator([(model1,60),(model2,40)])