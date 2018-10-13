import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from PIL import Image
import argparse

from models import *
from utils import *


# All implemented models
models = ['resnet', 'vgg16', 'googlenet', 'resnext', 'SimpleResNeXt_v1', 'SimpleResNeXt_v2']
parser = argparse.ArgumentParser()
parser.add_argument
parser.add_argument("-model", "--model-name", help="model name", dest="model_name", default='SimpleResNeXt_v1', choices=(tuple(models)))
args = parser.parse_args()

model_name = args.model_name

# Cifar-10 labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Use GPU or not
USE_GPU = True if torch.cuda.is_available() else False

# Preprocessing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load Model
net = load_model(model_name, USE_GPU)

# Where are the images & Read them
img_folder = './test_imgs/'
imgs = []
for img_path in glob.glob(img_folder+'*.png'):
	img = Image.open(img_path)
	imgs.append(img)

# predict each image
for i, img in enumerate(imgs):
	mat = np.asarray(img)
	x = transform_test(mat)
	x = x.unsqueeze(0)

	if USE_GPU:
		x = x.cuda()
	with torch.no_grad():
		net.eval()
		out = net(x)
		top_5_score, top_5_label = out.topk(5)
		top_5 = [classes[l] for i, l in enumerate(top_5_label.data.cpu().numpy()[0])] 
		title = [str(i+1) + ' ' + label + '\n' for i, label in enumerate(top_5)]
		title = ''.join(title)
		plt.subplot(2, 5, i+1), plt.imshow(img), plt.title(title), plt.yticks([]), plt.xticks([])
	if i > 10 :
		break
plt.show()





