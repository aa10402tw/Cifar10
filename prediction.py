import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import pickle
import argparse
import time
import json
import os
from PIL import Image

from models import *
from utils import *


# All implemented models
models = ['resnet', 'vgg16', 'googlenet', 'resnext']
model_name = 'resnext'

# Cifar-10 labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Use GPU or not
USE_GPU = True if torch.cuda.is_available() else False

# Batch size of train loader
BATCH_SIZE = 128     
     
# Preprocessing

transform_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])




# # Load Model
net = load_model(model_name, USE_GPU)

# # Where are the images & Read them
# img_folder = './test_imgs/'
# imgs = []
# # for img_path in glob.glob(img_folder+'*.jpg'):
# # 	img = Image.open(img_path)
# # 	imgs.append(img)

# # for img in imgs:
# # 	x = np.asarray(img)
# # 	x = transform_test(x)
# # 	x = np.expand_dims(x, axis=0)
# # 	x = torch.from_numpy(x)
# # 	if USE_GPU:
# # 		x = x.cuda()
# # 	print(x[0].shape)
# # 	print(x[0])
# # 	break
# 	# with torch.no_grad():
# 	# 	out = net(x)
# 	# 	top_5_score, top_5_label = out.topk(5)
# 	# 	top_5 = [classes[l] for l in top_5_label.data.cpu().numpy()[0]] 
# 	# 	print(top_5_score)
# 	# 	print(top_5)
# 	# 	plt.imshow(img)
# 	# 	plt.show()
		

# #Preprocessing
transform_default = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_normalize = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_toImg = transforms.Compose([
    transforms.ToPILImage(),
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_default) 
testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False) 
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


for (i, data), data_test in zip(enumerate(data_loader), test_loader):
	x, y = data
	x_test, y_test = data_test
	# print('--Data', x[0].mean().item())
	# print('--Test', x_test[0].mean().item())

	img = transform_toImg(x[0])
	mat = np.asarray(img)
	# print('Save Mat', mat.mean())
	img.save('./test_imgs/%s.png'%str(i))
	# print('Before', transform_test(mat).mean().item())

	with torch.no_grad():
		net.eval()
		x_test = x_test.cuda()
		save = x_test[0:1]
		print(x_test[0].mean().item())
		out = net(x_test)
		print(out[0])
		_, l = out[0].topk(1)

		print('pred', classes[l[0]])
		# print('real', classes[y_test[0]])
	break

	if i > 10:
		break

# 	x = x[0].data.numpy()
# 	x = np.transpose( x, (1,2,0) )
# 	label = classes[y[0]]
# 	x = (x).astype(np.float32)
# 	# print("Save", x[0][0])
# 	cv2.imwrite('./test_imgs/%s.jpg'%str(i), x, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
# 	break
# 	if i >= 10 :
# 		break

print('\n----\n')
for i in range(5):
	img = Image.open('./test_imgs/%s.png'%str(i))
	mat = np.asarray(img)
	# print('Read Mat', mat.mean())
	x = transform_test(mat)
	x = x.unsqueeze(0)
	
	# print("After", x.mean().item())
# 	# print(x.shape)
	with torch.no_grad():
		net.eval()
		x = x.cuda()
		save2 = x
		print(x.mean().item())
		out = net(x)
		print(out.shape)
		_, l = out[0].topk(1)
		print('pred', classes[l[0]])
		top_5_score, top_5_label = out.topk(5)
		top_5 = [classes[l] for l in top_5_label.data.cpu().numpy()[0]] 
		title = [label + '\n' for label in top_5]
		title = ''.join(title)
		plt.subplot(1, 5, i+1), plt.imshow(img), plt.title(title), plt.yticks([]), plt.xticks([])
plt.show()

print(save.mean().item())
print(save2.mean().item())

with torch.no_grad():
	out = net(save)
	out2 = net(save2)
	print(out)
	print(out2)
	_, l = out.topk(1)
	_, l2 = out.topk(1)
	print('pred', classes[l[0]])
	print('pred2', classes[l2[0]])