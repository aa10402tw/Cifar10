import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

VGG_architecture = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):

    def __init__(self, layers_name):
        super(VGG, self).__init__()
        self.featur_extractor = self._make_layers(layers_name)
        self.classifier = nn.Linear(512, 10)
        
    def _make_layers(self, layers_name):
        layers = []
        in_channels = 3
        for layer_name in layers_name:
            if layer_name == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = int(layer_name)
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)]
                in_channels = out_channels
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.featur_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def VGG13():
    return VGG(VGG_architecture['VGG13'])

def VGG16():
    return VGG(VGG_architecture['VGG16'])

def VGG19():
    return VGG(VGG_architecture['VGG19'])  

def test_VGG():
    net = VGG16()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test_VGG()

