import torch
import torch.nn as nn
import torch.nn.functional as F

# Default: BN before ReLU(dosen't make sense) 
# next try : pre-activation (BN->ReLU->Conv)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        
        # 如果 in_channels 與 out_channels 維度不同， 用 1x1 conv 讓維度相同
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):

        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = F.relu(out) 
        out = self.conv2(out)
        # Short cut
        residual = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out += residual
        # Try : BN after ReLU
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_blocks_list, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.layer1 = self.make_layer(ResidualBlock, out_channels=64,  num_blocks = num_blocks_list[0], stride=1) 
        self.layer2 = self.make_layer(ResidualBlock, out_channels=128, num_blocks = num_blocks_list[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, out_channels=256, num_blocks = num_blocks_list[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, out_channels=512, num_blocks = num_blocks_list[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #if num_blocks = 2, stride=2, strides = [2,1] (reduce dim at first block)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.conv1(x)    # (3, 32, 32)   -> (64, 32, 32)
        out = self.layer1(out) # (64, 32, 32)  -> (64, 32, 32)
        out = self.layer2(out) # (64, 32, 32)  -> (128, 16, 16)
        out = self.layer3(out) # (128, 16, 16) -> (256, 8, 8)
        out = self.layer4(out) # (256, 8, 8)   -> (512, 4, 4)
        out = F.avg_pool2d(out, kernel_size=(4,4)) # (512, 4, 4) -> (512, 1, 1)      
        out = out.view(out.size(0), -1) # (512, 1, 1) -> (512, 1)
        out = self.fc(out)     # (512, 1) -> (10, 1)
        return out


def ResNet18():
    return ResNet(ResidualBlock, num_blocks_list=[2,2,2,2])

def ResNet34():
    return ResNet(ResidualBlock, num_blocks_list=[3,4,6,3])

def ResNet50():
    return ResNet(ResidualBlock, num_blocks_list=[3,4,6,3])

def ResNet101():
    return ResNet(ResidualBlock, num_blocks_list=[3,4,23,3])

def ResNet152():
    return ResNet(ResidualBlock, num_blocks_list=[3,8,36,3])





def test_ResNet():
    net = ResNet18()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test_ResNet()



