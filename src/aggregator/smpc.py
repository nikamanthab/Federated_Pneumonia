import torch
import models
import random
from copy import deepcopy
from modelloader import getModelArchitecture
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

def normalize_weights(weights_array, fraction):
    for idx in range(len(weights_array)):
        weights_array[idx] = weights_array[idx]*fraction
    return weights_array

# def initialize_empty_models(n, serverargs):
#     '''
#         input: number of nodes (n)  and serverargs for architecture
#         output: generates a mdoel list with n tuples and number of samples as 1
#     '''
#     model = getModelArchitecture(serverargs)
#     initialize_empty_models = []
#     for _ in range(n):
#         instance_mdoel = deepcopy(model)
#         empty_model_list.append((instance_mdoel, 1))
#     return initialize_empty_models

def layer_sharing(model_list, serverargs):
    '''
        input: model list tuple and serverargs
        output: shuffled model list
    '''

    shuffling_matrix = []
    # Calling initialization models
    # shuffling_model_list = initialize_empty_models(len(model_list))
    '''
        Generating the shuffling matrix
        generates a matrix of dimension no_of_nodes*no_of_layers
        Matrix values are the destination location of the particular node layer
    '''
    num_of_nodes = len(model_list)
    num_of_layers = len(model_list[0][0].state_dict())
    for i in range(num_of_nodes):
        l = [0 for j in range(num_of_layers)]
        shuffling_matrix.append(l)

    for layer_num in range(num_of_layers):
        model_set = set(range(len(model_list)))
        for node_num in range(num_of_nodes):
            random_node = random.sample(model_set, 1)[0]
            model_set.remove(random_node)
            shuffling_matrix[node_num][layer_num] = random_node

    # for model, samples in model_list:
    #     shuffling_layers = []
    #     layers = list(range(len(model.state_dict())))
    #     while(len(layers)):
    #         random_index = random.randrange(0, len(layers))
    #         shuffling_layers.append(layers[random_index])
    #         layers.pop(random_index)
    #     import pdb; pdb.set_trace()
    #     shuffling_matrix.append(shuffling_layers)
    
    # Calling the matrix print function
    print_matrix(shuffling_matrix)

    # Creating model's array
    models_array = []
    total_samples = sum([samples for (model, samples) in model_list])
    for model, samples in model_list:
        fraction = samples/total_samples
        models_array.append(normalize_weights(weights_to_array(model), fraction))
        # models_array.append(weights_to_array(model))

    shuffling_models_array = deepcopy(models_array)

    shuffling_model_list = []
    for model_num in range(len(model_list)):
        for layer_num in range(len(model_list[0][0].state_dict().items())):
            destination_model = shuffling_matrix[model_num][layer_num]
            shuffling_models_array[destination_model][layer_num] = models_array[model_num][layer_num]
    
    for model_num in range(len(shuffling_models_array)):
        model_state = OrderedDict()
        model = getModelArchitecture(serverargs)
        for idx, key in enumerate(model.state_dict().keys()):
            model_state[key] = shuffling_models_array[model_num][idx]
        model.load_state_dict(model_state)
        shuffling_model_list.append((model, 1)) 

    return shuffling_model_list
    
def print_matrix(shuffling_matrix):
    '''
        input: shuffling matrix
        Print the 2D matrix
    '''
    print("Shuffling matrix - rows represent nodes, columns represent layers")
    for i in shuffling_matrix:
        for j in i:
            print(j, end=' ')
        print()