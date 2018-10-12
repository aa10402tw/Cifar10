import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import pickle
import time
import json
import os

from models import *

def create_model(model_name, USE_GPU):
    if model_name == 'resnext':
        net = ResNeXt29_4x64d()


    elif model_name == 'resnet':
        net = ResNet18()

    elif model_name == 'vgg16':
        net = VGG16()
        
    elif model_name == 'googlenet':
        net = GoogLeNet()

    elif model_name == 'SimpleResNeXt_v1':
        net = SimpleResNeXt_v1()
        
    elif model_name == 'SimpleResNeXt_v2':
        net = SimpleResNeXt_v2()

    else:
        raise "There is no model called %s"%(model_name)  

    if USE_GPU:
        net = net.cuda()
    return net

def load_model(model_name, USE_GPU):
    net = create_model(model_name, USE_GPU)
    net.load_state_dict(torch.load('./trained_model/%s.pkl'%(model_name)))
    return net

def top_k_acc(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_model(net, train_loader, test_loader, criterion, optimizer, num_epochs, model_name=None, save_best=True, USE_GPU=True): 
    model_info = read_json()
    best_acc = model_info[model_name]['test_acc_top1'] if save_best else 100
    best_acc = 0 if best_acc is None else best_acc
    history = {'acc':[],  'test_acc_top1':[], 'test_acc_top5':[]}
    
    for epoch in range(num_epochs):
        # Train 
        net.train()
        # Adjust Learning Rate 
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * 0.95

        # if epoch == 150:
        #     optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # elif epoch == 250:
        #     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        
        correct, total, sum_loss = 0, 0, 0
        pbar = tqdm(total=len(train_loader), unit=' batches', ncols=100)
        pbar.set_description('Epoch %i/%i (Training)' % (epoch+1, num_epochs))
        for i, data in enumerate(train_loader):
            x, y = data
            if USE_GPU:
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            # forward
            outputs = net(x)
            loss = criterion(outputs, y)
            # backprop
            loss.backward()
            optimizer.step()
            # Compute correct
            _, predicted = torch.max(outputs.data.cpu(), 1)  
            total += int(y.data.cpu().size(0))
            sum_loss += float(loss.data.cpu().item())
            correct += int(predicted.eq(y.data.cpu()).sum().item())
            pbar.set_postfix( {'loss': '%.2f'%(sum_loss/(i+1)), 'acc': '%.2f'%(correct/total) } )
            pbar.update()
            
        # # Record History
        history['acc'] = history['acc'] + [correct/total]
        pbar.close()
        time.sleep(1)

        # Test
        net.eval() # Dropout and BatchNorm (and maybe some custom modules) behave differently during training and evaluation
        with torch.no_grad():
            correct_top1, correct_top5, total, sum_loss = 0, 0, 0, 0
            pbar = tqdm(total=len(test_loader), unit=' batches', ncols=30)
            pbar.set_description('Epoch %i/%i (Testing)' % (epoch+1, num_epochs))
            for i, data in enumerate(test_loader):
                x, y = data
                if USE_GPU:
                    x, y = x.cuda(), y.cuda()
                # forward
                outputs = net(x)
                loss = criterion(outputs, y)
                # Compute correct
                _, predicted = torch.max(outputs.data.cpu(), 1)

                topk = (1,5)
                maxk = max(topk)
                target = y
                batch_size = 100
                _, pred = outputs.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                res = []
                for k in topk:
                    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                    res.append(correct_k.mul_(100.0 / batch_size))
                top_1, top_5 = res
                correct_top1 +=  top_1
                correct_top5 += top_5
               
                total += int(y.data.cpu().size(0))
                pbar.set_postfix( {'top1_acc': '%.2f'%(correct_top1/total), 'top5_acc': '%.2f'%(correct_top5/total) } )
                pbar.update()
                
            # Record History
            history['test_acc_top1'] = history['test_acc_top1'] + [(correct_top1/total).item()]
            history['test_acc_top5'] = history['test_acc_top5'] + [(correct_top5/total).item()]
            pbar.close()
            time.sleep(1)

        # Save model if perform better 
        if (correct_top1/total) > best_acc:
            best_acc = correct_top1/total
            model_info = read_json()
            model_info[model_name] = {'num_epochs':epoch, 'test_acc_top1':(correct_top1/total).item(), 'test_acc_top1':(correct_top5/total).item()}
            write_json(model_info)
            torch.save(net.state_dict(), './trained_model/%s.pkl'%(model_name))
        save_hisotry(model_name, history)
    return history


def read_json():
    with open('./trained_model/models_info.json', 'r') as json_file:  
        data = json.load(json_file)
        model_info = data
    return model_info

def write_json(model_info):
    with open('./trained_model/models_info.json', 'w') as outfile:
        json.dump(model_info, outfile)

def init_model_info(models):
    model_info = {}
    for model in models:
        model_info[model] = {'num_epochs':None, 'test_acc_top1':None, 'test_acc_top5':None}
    return model_info

def init(models):
    os.makedirs('trained_model')
    model_info = init_model_info(models)
    write_json(model_info)


###############
###   plot  ###
###############
def show_train_history(history, train, test_top1, test_top5):
    plt.plot(history[train])
    plt.plot(history[test_top1])
    plt.plot(history[test_top5])
    plt.title('Train History')
    plt.ylim([0,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test_top1', 'test_top5'], loc='lower right')
    plt.show()
    
def save_hisotry(model_name, history):
     with open('./trained_model/histories.pkl', 'wb+') as f:
            try:
                histories = pickle.load(f)
            except EOFError:
                histories = {}  # or whatever you want
            histories[model_name] = history
            pickle.dump(histories, f)
    
def load_history(model_name=None):
    with open('./trained_model/histories.pkl', 'rb') as f:
        histories = pickle.load(f)
        if model_name is not None:
            return histories[model_name]
        else:
            return histories


if __name__ == '__main__':
    history = load_history(model_name='myresnext')
    show_train_history(history, 'acc', 'test_acc_top1', 'test_acc_top5')

