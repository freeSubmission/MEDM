######################################################################################
#    MEDM： Minimal Entropy Diversity Maximization for Unsupervised Domain Adaptation
#    DataSet: VisDA Classification Challenge
#    Please specify a domain transferring task: source_name -->  target_name
#    Date: created on Jan. 18, 2019
#          updated on Jun. 18, 2019:
#          1) SGD optimizer -->Adam optimizer
#          2) class-weighted loss --> standard cross entropy loss without weighting
#    All rights reserved, please report any debug problem to blindreview@163.com
######################################################################################
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
import numpy as np
from torch.utils import model_zoo
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
#import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

# Training settings
num_experiments = 1
batch_size = 32
num_classes = 12
epochs = 10
lr = 0.0001
momentum = 0.9
no_cuda =False
seed = 999
log_interval = 1000
l2_decay = 5e-4
root_path = "/home/xfuwu/data/VisDA/" #Please replace your root_path for VisDA dataset
#root_path = "root_path_to_VisDA/"  #Please replace your root_path for VisDA dataset
source_name = "train"
target_name = "validation"

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 4, 'pin_memory': False} if cuda else {}

source_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)

def load_pretrain(model):
    url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    #url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if not "cls_fc" in k  and  not "num_batches_tracked" in k:
            model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    model.load_state_dict(model_dict)
    return model

#computing the label distributions of both source and target domains
def label_dist_dataset():
    label_dist_src = list(np.zeros(num_classes))
    source_count_loader = data_loader.load_counting(root_path, source_name, 512, kwargs)

    for source_data, source_label in source_count_loader:
        for cls in list(source_label):
            label_dist_src[cls] += 1.0

    label_dist_tgt = list(np.zeros(num_classes))
    target_count_loader = data_loader.load_counting(root_path, target_name, 512, kwargs)

    for target_data, target_label in target_count_loader:
        for cls in list(target_label):
            label_dist_tgt[cls] += 1.0

    return (label_dist_src, label_dist_tgt)


def train(epoch, model):
    LEARNING_RATE = lr
    print("learning rate：", LEARNING_RATE)

    ##1. update with SGD-->Adam---------------------------------------------------------
    # optimizer = torch.optim.SGD([
    #     {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
    # ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

    optimizer = torch.optim.Adam([
        {'params': model.sharedNet.parameters(), 'lr': LEARNING_RATE / 100},
        {'params': model.cls_fc.parameters(),'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE, weight_decay=l2_decay)

    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_train_loader)

    i = 1
    while i <= len_source_loader:
        model.train()
        source_data, source_label = data_source_iter.next()

        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        clabel_src = model(source_data)  #softmax forms
        ## Source classification loss
        ## 2. update with class-weighted loss ---> standard cross entropy loss--------------------
        #label_loss = (ws_batch * F.nll_loss(clabel_src.log(), source_label, reduce=False)).mean()
        label_loss = F.nll_loss(clabel_src.log(), source_label)

        target_data, target_label = data_target_iter.next()
        if i % len_target_loader == 0:
            data_target_iter = iter(target_train_loader)
        if cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()
        target_data = Variable(target_data)
        clabel_tgt = model(target_data)

        ## Target category diversity loss
        pb_pred_tgt = clabel_tgt.sum(dim=0)
        pb_pred_tgt = 1.0/pb_pred_tgt.sum() * pb_pred_tgt  #normalizatoin to a prob. dist.
        target_div_loss=  - torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-6)))

        target_entropy_loss= -torch.mean((clabel_tgt * torch.log(clabel_tgt + 1e-6)).sum(dim=1))
        total_loss = label_loss + 1.0 * target_entropy_loss - 0.2 * target_div_loss

        ##1: Training shared network and label classifier
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tentropy_Loss: {:.6f}\tlabel_Loss: {:.6f}\tdiv_Loss: {:.6f}'.format(
                epoch, i * len(source_data),len_source_dataset,
                100. * i / len_source_loader, target_entropy_loss.data[0], label_loss.data[0], target_div_loss.data[0]))
        i = i + 1

def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    Dict_all = list(np.zeros(num_classes))
    Dict_acc = list(np.zeros(num_classes))
    """
    Dict_name = {0: 'aeroplane', 1: 'bicycle', 2: 'bus', 3: 'car', 4: 'horse', 5: 'knife', 6: 'motorcycle', 7: 'person', \
                 8: 'plant', 9: 'skateboard', 10: 'train', 11: 'truck'}
    """
    for target_data, target_label in target_test_loader:
        target_label = target_label.long()
        if cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()
        target_data, target_label = Variable(target_data), Variable(target_label)
        out_tgt = model(target_data)  #prob
        test_loss += F.nll_loss(out_tgt.log(), target_label, size_average=False).data[0] # sum up batch loss
        pred = out_tgt.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target_label.data.view_as(pred)).cpu().sum()
        target_label = target_label.data.cpu()

        pred = pred.data.cpu()
        for j in range(target_label.numpy().shape[0]):
            Dict_all[target_label[j].item()] += 1
            if target_label[j] == pred[j]:
                Dict_acc[pred[j].item()] += 1
    test_loss /= len_target_dataset
    for j in range(len(Dict_all)):
        Dict_acc[j] = Dict_acc[j] / Dict_all[j] * 100.

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        target_name, test_loss, correct, len_target_dataset,
        100. * correct / len_target_dataset))
    print('aeroplane:', Dict_acc[0], 'bicycle:', Dict_acc[1], 'bus:', Dict_acc[2], 'car:', Dict_acc[3], 'horse:', Dict_acc[4],
          'knife:', Dict_acc[5], 'motorcycle:', Dict_acc[6], 'person:', Dict_acc[7], 'plant:',Dict_acc[8], 'skateboard:',
          Dict_acc[9], 'train:', Dict_acc[10], 'truck:',Dict_acc[11])
    all_acc =0
    for i in range(len(Dict_acc)):
        all_acc += Dict_acc[i]
    average_acc = all_acc/12
    print('average_acc',average_acc)
    return correct, average_acc ,Dict_acc

def one_hot(label):
    size = label.size(0)
    label = label.data.cpu().numpy()
    label = torch.from_numpy(np.reshape(label,(size,1)))
    one_hot_label = torch.zeros(size, 12).scatter_(1, label, 1)
    one_hot_label = Variable(one_hot_label.cuda())

    return one_hot_label

if __name__ == '__main__':

    ws, wt = label_dist_dataset()
    #print('label_dist_src',ws)
    #print('label_dist_tgt',wt)
    wsmax= max(ws)
    wtmax = max(wt)

    sum_wt = sum(wt)
    pt_true = [x/sum_wt for x in wt]
    print('true cls-dist:', pt_true)

    #ws = [wsmax/x for x in ws]
    #wt = [wtmax/x for x in wt]

    true_accuracy = list(np.zeros(num_experiments))
    true_acc_max = list(np.zeros(num_experiments))
    avg_accuracy = list(np.zeros(num_experiments))
    avg_acc_max = list(np.zeros(num_experiments))
    class_list= list(np.zeros(num_experiments))

    for ex in range(num_experiments):
        model = models.MEDM(num_classes=12)
        print (model)
        if cuda:
            model.cuda()
        model = load_pretrain(model)
        t_correct_true = 0
        t_correct_average = 0

        for epoch in range(1, epochs + 1):
            train(epoch, model)
            true_correct, average_acc, Dict_acc = test(model)

            if true_correct > t_correct_true:
                t_correct_true = true_correct
            if average_acc > t_correct_average:
                t_correct_average = average_acc
            print('Experi-No: {} source: {} to target: {} max correct: {} max true accuracy{: .2f}%  max average accuracy{: .2f}%\n'.format(
                  ex+1, source_name, target_name, t_correct_true, 100. * t_correct_true / len_target_dataset ,t_correct_average))
        true_accuracy[ex] = 100. * true_correct/len_target_dataset
        true_acc_max[ex] = 100. * t_correct_true/len_target_dataset
        avg_accuracy[ex] = 100. * average_acc/len_target_dataset
        avg_acc_max[ex] = 100. * t_correct_average / len_target_dataset
        class_list[ex] = Dict_acc

    print ('True Accs:',true_accuracy)
    print ('Avg Accs:',avg_accuracy)
    print ('True Max-Accs:', true_acc_max)
    print ('Avg Max-Accs:',avg_acc_max)
    true_avg_acc = sum(true_accuracy)/len(true_accuracy)
    avg_avg_acc = sum(avg_accuracy)/len(avg_accuracy)
    print('True Avg-Acc:', true_avg_acc)
    print('Avg-Acc:', avg_avg_acc)
    print ('class acc:',class_list)