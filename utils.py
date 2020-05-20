import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import torch
import random

def reset_seed(device, x=0):
    # Set all seeds
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(x)

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def smooth_plot(x, y=None, label='', halflife=10):
  if y is None:
    y_int = x
  else:
    y_int = y
  x_ewm = pd.Series(y_int).ewm(halflife=halflife)
  color = next(plt.gca()._get_lines.prop_cycler)['color']
  if y is None:
    plt.plot(x_ewm.mean(), label=label, color=color)
    #plt.plot(y_int, color=color, alpha=0.3)
    plt.fill_between(np.arange(x_ewm.mean().shape[0]), x_ewm.mean() + x_ewm.std(), x_ewm.mean() - x_ewm.std(), color=color, alpha=0.3)
  else:
    plt.plot(x, x_ewm.mean(), label=label, color=color)
    plt.fill_between(x, y_int + x_ewm.std(), y_int - x_ewm.std(), color=color, alpha=0.3)

def l2_regularization(net, C):
     l2 = 0
     for p in net:
      l2 = l2 + p.square().sum()
     return C * l2

def standard_train_step(net, loss_fn, opt, xb, yb, C=0):
    net.train()
    y_pred = net(xb)
    loss_epoch = loss_fn(y_pred, yb) + l2_regularization(net.parameters(), C=C)
    loss_epoch.backward()
    opt.step()
    opt.zero_grad()
    return loss_epoch.item()

def evaluate_net(net, test_loader, test_metric, device):
    with torch.no_grad():
        test_i = 0.0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            test_i += test_metric(yb, net(xb))
    return test_i / len(test_loader.dataset)