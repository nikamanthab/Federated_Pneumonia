import models
import torch

''' ***Include additional model architectures here*** '''
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
    Input: arguments for server
    Output: torch model
    '''
    model = switcher.get(serverargs['architecture'], "architecture name mismatch - check help for architectures")
    model = model()
    torch.save(model, serverargs['aggregated_model_location']+'agg_model.pt')
    return True

def getModelArchitecture(serverargs):
    model = switcher.get(serverargs['architecture'], "architecture name mismatch - check help for architectures")
    model = model().to(serverargs['device'])
    return model

def loadModel(location):
    '''
    Loads the model and returns
    Input: model location
    Output: model
    '''
    return torch.load(location)

    