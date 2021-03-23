import models
import torch

switcher = { 
    "TwoLayerNet": models.TwoLayerNet,
    "ResNeXt50": models.ResNeXt50,
    "ResNet18": models.ResNet18,
    "AlexNet": models.Alex,
    "VGGNet": models.VGGNet,
    "Inceptionv3Net": models.Inceptionv3Net,
    "GoogleNet": models.GoogleNet
} 

def createInitialModel(serverargs):
    '''
    ***Include additional model architectures here***
    Input: arguments for server
    Output: torch model
    '''    
    model = switcher.get(serverargs['architecture'], "architecture name mismatch - check help for architectures")
    model = model()
    torch.save(model, serverargs['aggregated_model_location']+'agg_model.pt')
    return True

def createRandomInitializedModel(serverargs):
    model = switcher.get(serverargs['architecture'], "architecture name mismatch - check help for architectures")
    return model()

def loadModel(location):
    '''
    Loads the model and returns
    Input: model location
    Output: model
    '''
    return torch.load(location)

    