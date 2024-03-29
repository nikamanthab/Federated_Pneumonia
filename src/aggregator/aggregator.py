import torch
import torch.optim as optim
import numpy as np
import random
from modelloader import getModelArchitecture, loadModel
from collections import OrderedDict

def weights_to_array(model):
    '''
        input: pytorch model
        output: array of tensors of model weights
    '''
    model_weights = []
    for (key, param) in model.state_dict().items():
    # for param in model.parameters():
        model_weights.append(param) # check without .data
    return model_weights

def geomed_weights_to_array(model, args):
    '''
        input: pytorch model
        output: array of tensors of model weights
    '''
    model_weights = []
    # for (key, param) in model.state_dict().items():
    for param in list(model.parameters())[args['train_layers']:]:
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
        temp = torch.zeros(node_weights[0][layer_idx].shape).to(args['device'])
        for node_idx in range(len(node_weights)):
            if args["smpc"]:
                fraction = 1
            else:
                fraction = (node_samples[node_idx]/total_no_samples)
            temp+= fraction*node_weights[node_idx][layer_idx]
        aggregated_weights.append(temp)
    agg_model = getModelArchitecture(args)
    
    agg_state = OrderedDict()
    for idx, key in enumerate(agg_model.state_dict().keys()):
        agg_state[key] = aggregated_weights[idx]
    agg_model.load_state_dict(agg_state)
#     agg_model = loadModel(args['aggregated_model_location']+'agg_model.pt').to(args['device'])
#     for idx, (key, param) in enumerate(agg_model.state_dict().items()):
#         agg_model.state_dict()[key] = aggregated_weights[idx]
#         import pdb; pdb.set_trace()
    return agg_model

# x = fed_avg_aggregator([(model1,60),(model2,40)])

#COMED aggregator
def comed_aggregator(model_data, args):
    '''
        input: array of tuples containing model 
        and the number of samples from each respective node
        output: COMED aggregated model
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
        temp = torch.zeros(node_weights[0][layer_idx].shape).to(args['device'])
        for node_idx in range(len(node_weights)):
            if(node_idx == 0):
                temp = torch.flatten(node_weights[node_idx][layer_idx]).unsqueeze(1)
            else:
                layer_flattened = torch.flatten(node_weights[node_idx][layer_idx]).unsqueeze(1)
                temp = torch.cat((temp, layer_flattened),1)
        temp = temp.detach().cpu().numpy()
        temp = np.median(temp,1)
        temp = torch.from_numpy(temp)
        temp = torch.reshape(temp, layer_shape)
        aggregated_weights.append(temp)

    agg_model = getModelArchitecture(args)

    agg_state = OrderedDict()
    for idx, key in enumerate(agg_model.state_dict().keys()):
        agg_state[key] = aggregated_weights[idx]
    agg_model.load_state_dict(agg_state)
    # for idx, param in enumerate(agg_model.parameters()):
    #     param.data = aggregated_weights[idx]
    return agg_model

def g(z, node_weights, node_samples, total_no_samples, device, args):
    '''
        Optimizer loss function for geometric median
        Refer equation 3 in Krishna Pillutla et al., Robust Aggregation for Federated Learning
        
        input:  z - aggregator weights to minimize
                node_weights - array of model weights from weights_to_array function
                node_samples - array of sample counts from each node
                total_no_samples - sum(node_samples)
        output: weighted summation of euclidean norm with respect to the aggregator weights        
    '''
    summation = 0.0 #torch.Tensor([0.0]).to(device)
    for layer_idx in range(len(node_weights[0])):
        temp = torch.zeros(node_weights[0][layer_idx].shape).to(device)
        for node_idx in range(len(node_weights)):
            euclidean_norm = (z[layer_idx] - node_weights[node_idx][layer_idx])**2
            if args["smpc"]:
                weight_alpha = 1
            else:
                weight_alpha = node_samples[node_idx]/total_no_samples
            temp = temp + (weight_alpha * euclidean_norm)
        summation = summation + sum(temp.flatten())
    return summation

#test line
#g(weights_to_array(agg_model), [weights_to_array(model1), weights_to_array(model2)], [5, 5], 10)

def optimizeGM(agg_model,  node_weights, node_samples, total_no_samples, args):
    optimizer = optim.Adam(list(agg_model.parameters())[args['train_layers']:], lr=args["agg_optim_lr"])
    for _ in range(args["agg_iterations"]):
        optimizer.zero_grad()
        loss = g(geomed_weights_to_array(agg_model, args), node_weights, node_samples, total_no_samples, args['device'], args)
        print(loss, type(loss))
        loss.backward()
        optimizer.step()
    return agg_model

def Geometric_Median(model_data, args):
    '''
        input: array of tuples containing model 
        and the number of samples from each respective node
        output: geometric median aggregated model
    '''
    total_no_samples = 0
    
    # creates an array of weights and sample counts
    # shape -> no_models*no_layers*dim_of_layer
    node_weights = []
    node_samples = []
    for model,no_samples in model_data:
        node_weights.append(geomed_weights_to_array(model, args))
        node_samples.append(no_samples)
    # calculates the total number of samples
        total_no_samples += no_samples
    # agg_model = getModelArchitecture(args)
    agg_model = comed_aggregator(model_data, args)
    a = []
    for i in agg_model.parameters():
        a.append(i.clone())
    agg_model = optimizeGM(agg_model, node_weights, node_samples, total_no_samples, args)
    b = []
    for i in agg_model.parameters():
        b.append(i.clone())
    for i in range(len(a)):
        print(torch.equal(a[i].data, b[i].data))
    return agg_model