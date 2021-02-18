import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class TwoLayerNet(nn.Module):
    def __init__(self):
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

class Alex(nn.Module):
    def __init__(self):
        super(Alex, self).__init__()
        self.model = torchvision.models.AlexNet(num_classes=2)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.model = torchvision.models.vgg11(pretrained=False, progress=True)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=1000, bias=True)
    
    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

class Inceptionv3Net(nn.Module):
    def __init__(self):
        super(Inceptionv3Net, self).__init__()
        self.model = torchvision.models.inception_v3(pretrained=False, progress=True)
#         self.model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    
    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.model = torchvision.models.googlenet(pretrained=False, progress=True)
        self.model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    
    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

class ResNeXt50(nn.Module):
    def __init__(self):
        super(ResNeXt50, self).__init__()
        self.model = torchvision.models.resnext50_32x4d(pretrained=False, progress=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         self.fc1 = nn.Linear(1000, 2)
        self.model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
#         self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.model(x)
        return x
#         x = self.softmax(x)
#         x = self.fc1(x)
#         return F.log_softmax(x, dim=1)

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False, progress=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
#         self.model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True),
#                                 nn.Linear(in_features=256, out_features=128, bias=True),
#                                 nn.Linear(in_features=128, out_features=2, bias=True))
    def forward(self,x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)
