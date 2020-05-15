# -*- coding: utf-8 -*-

from sklearn import datasets, preprocessing, model_selection
import torch
from torch.utils import data
import torchvision

def load_boston():
    # Boston
    dataset = datasets.load_boston()
    X, y = dataset['data'], dataset['target']
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = preprocessing.MinMaxScaler().fit_transform(y.reshape(-1, 1))
  
    Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y)
  
    Xtrain = torch.from_numpy(Xtrain).float()
    Xtest = torch.from_numpy(Xtest).float()
    ytrain = torch.from_numpy(ytrain).float()
    ytest = torch.from_numpy(ytest).float()
    
    train_data = data.TensorDataset(Xtrain, ytrain)
    test_data = data.TensorDataset(Xtest, ytest)
    
    return data.DataLoader(train_data, batch_size=32, shuffle=True),\
      data.DataLoader(test_data, batch_size=32, shuffle=True)
      

def load_mnist():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=32, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=32, shuffle=False)
    return train_loader, test_loader