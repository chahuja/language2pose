import torch

class Loss():
  def __init__(self, losses, kwargs_list):
    self.losses = losses
    self.kwargs_list = kwargs_list
    self.loss_list = []
    for loss, kwargs in zip(self.losses, self.kwargs_list):
      if hasattr(torch.nn, loss):
        loss = 'torch.nn.' + loss 
      self.loss_list.append(eval(loss)(**kwargs))

  def __call__(self, y_cap, y):
    loss = 0
    for loss_fn in self.loss_list:
      loss += loss_fn(y_cap, y)

    return loss
