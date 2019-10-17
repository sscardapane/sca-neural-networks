# -*- coding: utf-8 -*-

from sklearn import datasets, preprocessing, model_selection

def load_boston():
  # Boston
  data = datasets.load_boston()
  X, y = data['data'], data['target']
  X = preprocessing.MinMaxScaler().fit_transform(X)
  y = preprocessing.MinMaxScaler().fit_transform(y).reshape(-1, 1)
  return model_selection.train_test_split(X, y)