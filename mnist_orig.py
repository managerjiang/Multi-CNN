# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:24:07 2022

@author: JX
"""


'''
系统环境: Windows10
Python版本: 3.7
PyTorch版本: 1.1.0
cuda: no
'''
import torch
import torch.nn.functional as F   # 激励函数的库
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time
import torch.nn as nn

# 定义全局变量
n_epochs = 9     # epoch 的数目
batch_size = 20  # 决定每次读取多少图片

# 定义训练集个测试集，如果找不到数据，就下载
dataset = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
test_data = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())


# 创建加载器
train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = 0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = 0)

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
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)
'''
# 建立一个四层感知机网络
class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super(MLP,self).__init__()    # 
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(784,512)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(512,128)  # 第二个隐含层
        self.fc3 = torch.nn.Linear(128,10)   # 输出层
        
    def forward(self,din):
        # 前向传播， 输入值：din, 返回值 dout
        din = din.view(-1,28*28)       # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout), dim=1)  # 输出层使用 softmax 激活函数
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return dout
'''
# 训练神经网络
def train():
    torch.manual_seed(1234)
    #定义损失函数和优化器
    lossfunc = torch.nn.CrossEntropyLoss()
    lr=0.01
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data,target in train_loader:
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            output = model(data)    # 得到预测值
            loss = lossfunc(output,target)  # 计算两者的误差
            loss.backward()         # 误差反向传播, 计算参数更新值
            #optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    d_p = p.grad				
                    p.add_(d_p, alpha=-lr)
            train_loss += loss.item()*data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        # 每遍历一遍数据集，测试一下准确率
        #test()

# 在数据集上测试神经网络
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    return 100.0 * correct / total

# 声明感知器网络
model = Net()

if __name__ == '__main__':
    start = time.time()
    train()
    end = time.time()
    runntime = end - start
    print('runtime:',runntime)
