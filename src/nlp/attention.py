import torch
import pdb

def dot(b1, b2, dim=-1):
  return (b1*b2).sum(dim=dim)

def add_eps(b, eps):
  mask = (b == 0).double()
  return b*(1-mask) + mask*(b + eps)

def batch_cosine_distance(b1, b2, dim=-1, eps=1e-10):
  b1 = add_eps(b1, eps)
  b2 = add_eps(b2, eps)
  return dot(b1, b2, dim=dim)/(dot(b1, b1, dim=dim)**(0.5) * dot(b2, b2, dim=dim)**(0.5))

def attention(w, h, mask):
  '''
  w: BxTxH
  h: BxH
  mask: BxT
  '''
  a = batch_cosine_distance(w, h.unsqueeze(1), dim=-1) * mask
  return (a.unsqueeze(-1)*w).sum(dim=1)/a.sum(dim=1).unsqueeze(-1)
