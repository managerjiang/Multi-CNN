# Multi-CNN
This work solves the large-scale classification problem over the MNIST dataset by a multilayer convolutional neural network (MCNN) model. 
With the help of PyTorch distributed-training tutorial, we apply three optimization algorithms (decentralized gradient descent algorithm (DenGD) in the tutorial, the traditional centralized gradient descent (CenGD) and the proposed distributed algorithm (DisGD))
 to solve the classification problem over the MNIST dataset.
 mnist_orig.py: CenGD optimization
 mnist_parl.py: DenGD optimization
 mnist_dis_vary.py: DisGD optimization
