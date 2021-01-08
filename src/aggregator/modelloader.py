import models
import torch
def createInitialModel(serverargs):
    '''
    ***Include additional model architectures here***
    Input: arguments for server
    Output: torch model
    '''
    switcher = { 
        "TwoLayerNet": models.TwoLayerNet,
        "ResNeXt50": models.ResNeXt50,
        "ResNet18": models.ResNet18
    } 
    model = switcher.get(serverargs['architecture'], "architecture name mismatch - check help for architectures")
    model = model()
    torch.save(model, serverargs['aggregated_model_location']+'agg_model.pt')
    return True

def loadModel(location):
    '''
    Loads the model and returns
    Input: model location
    Output: model
    '''
    return torch.load(location)

    