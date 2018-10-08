import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import argparse
import time
import json
import os

from models import *
from utils import *

# All implemented models
models = ['resnet', 'vgg16', 'googlenet', 'resnext']

# set up for first time
if not os.path.isdir('trained_model'):
    init(models)

# Cifar-10 labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Use GPU or not
USE_GPU = True if torch.cuda.is_available() else False

# Batch size of train loader
BATCH_SIZE = 128     
     
# Preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # padding with zero and crop to 32*32
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # mean & std
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)   

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# Create training
parser = argparse.ArgumentParser()
parser.add_argument
parser.add_argument("-model", "--model-name", help="model name", dest="opt", default="resnext", choices=(tuple(*models)))
args = parser.parse_args()

model_name = args.model
net = create_model(model_name, USE_GPU)

# set Loss and Optimizer
LR = 0.1 
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

history = train_model(net, train_loader, test_loader, criterion, optimizer, num_epochs=350, model_name=model_name, save_best=True, USE_GPU=USE_GPU)

print('Bset Acc:', max(history['test_acc']))

show_train_history(history, 'acc', 'test_acc')
show_train_history(history, 'loss', 'test_loss')