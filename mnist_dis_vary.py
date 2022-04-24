 # -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:24:13 2022

@author: JX
"""


import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io as scio
import numpy as np
import time
class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
'''   
##感知机模型

# 建立一个四层感知机网络
class MLP(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self):
        super(MLP, self).__init__()  #
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(784, 512)  # 第一个隐含层
        self.fc2 = torch.nn.Linear(512, 128)  # 第二个隐含层
        self.fc3 = torch.nn.Linear(128, 10)  # 输出层

    def forward(self, din):
        # 前向传播， 输入值：din, 返回值 dout
        din = din.view(-1, 28 * 28)  # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))  # 使用 relu 激活函数
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout), dim=1)  # 输出层使用 softmax 激活函数
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return dout
'''
#卷积神经网络
class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))

#train_loader = torch.utils.data.DataLoader(dataset, batch_size = 20, num_workers = 0)
test_loader = torch.utils.data.DataLoader(dataset, batch_size = 20, num_workers = 0)

def partition_dataset():
    """ Partitioning MNIST """
    '''
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    '''
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=int(bsz), shuffle=True)

    return train_set, bsz
'''
##梯度平均函数
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
'''
##切换拓扑邻接矩阵
def lambda_f(C_store,epo,size):
    num=(epo+1)//5
    yu=(epo+1)%5
    if num==0:
        y=np.eye(size)
    else:
        y=(C_store[0][0]@C_store[0][1]@C_store[0][2]@C_store[0][3]@C_store[0][4])**num
    if yu==0:
        ry=y
    elif yu==1:
        ry=y@C_store[0][0]
    elif yu==2:
        ry=y@C_store[0][0]@C_store[0][1]
    elif yu==3:
        ry=y@C_store[0][0]@C_store[0][1]@C_store[0][2]
    elif yu==4:
        ry=y@C_store[0][0]@C_store[0][1]@C_store[0][2]@C_store[0][3]
    return ry
##测试函数
def test(model):
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Rank ', dist.get_rank(), ', epoch ', 'testAccuracy: %d %%' % (
                  100 * correct / total))
    return 100.0 * correct / total
##分布式训练
def run(rank, size):
    """ Distributed Synchronous SGD Example """
    dataFile = 'F:/anaconda_spyder/MLP/data/C_sw_admat.mat' #C_ad_matrix  C_sw_admat.mat
    data = scio.loadmat(dataFile)
    C_store = data['C_store'] #C = data['a']
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
#    model = model.cuda(rank)
    lr=0.01
    optimizer = optim.SGD(model.parameters(), lr=0.01) #momentum=0.5
    lossfunc = torch.nn.CrossEntropyLoss()  ############
    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(9):
        epoch_loss = 0.0
        C = lambda_f(C_store, epoch,size)
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
#            data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
            optimizer.zero_grad()
            output = model(data)
            loss = lossfunc(output, target)
            epoch_loss += loss.item()
            loss.backward()
            #optimizer.step()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    d_p = p.grad
                    p.add_(d_p, alpha=-lr)
        print('Rank ', rank, ', epoch ',epoch, ': ', epoch_loss / num_batches)
        for p in model.parameters():
            sum_d=torch.zeros_like(p.data, dtype=torch.float)
            req = None
            req1=None
            for j in range(size):
               if (rank!=j) & (C[rank][j] != 0) :
                   req=dist.isend(tensor=p.data, dst=j)
                   neigh_tensor=torch.zeros_like(p.data, dtype=torch.float)
                   req1=dist.irecv(tensor=neigh_tensor,src=j)
                   req1.wait()
                   sum_d += C[rank][j] * neigh_tensor
                   req.wait()
            sum_d+=C[rank][rank]*p.data
            p.data=np.sign(sum_d)*np.maximum(np.abs(sum_d)-lr,0)
        #test(model)

def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
##主程序

if __name__ == "__main__":
    size = 3
    processes = []
    start = time.time()
    #mp.set_start_method("spawn")
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    end = time.time()
    runntime = end - start
    print('runtime:',runntime)