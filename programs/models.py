import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class TwoLayerNet(nn.Module):
    def __init__(self, device):
        super(TwoLayerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class AddLayer(nn.Module):
    def __init__(self, device):
        super(AddLayer, self).__init__()
        self.fc1 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# def ResNeXt50(device):
#     resnext = torchvision.models.resnext50_32x4d(pretrained=False, progress=True)
#     resnext.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     net = AddLayer(device=device)
#     model = nn.Sequential(
#         resnext, net
#     )
#     return model


class ResNeXt50(nn.Module):
    def __init__(self, device):
        super(ResNeXt50, self).__init__()
        self.model = torchvision.models.resnext50_32x4d(pretrained=False, progress=True).to(device=device)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc1 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)