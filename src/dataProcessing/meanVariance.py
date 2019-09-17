import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataUtils import *

from pycasper.name import Name
from pycasper.BookKeeper import *
from argsUtils import argparseNloop

import numpy as np
from tqdm import tqdm
import pdb

def loadMeanVariance(columns_subset, seed, mask, feats_kind, dataset, f_new):
  mask_str = '_'.join(['{}'.format(i) for i in mask])
  mean = pd.read_csv('dataProcessing/meanVar/mean_{}_{}_{}_{}_{}.csv'.format(seed, mask_str, feats_kind, dataset, f_new), index_col=0)[columns_subset].values
  variance = pd.read_csv('dataProcessing/meanVar/variance_{}_{}_{}_{}_{}.csv'.format(seed, mask_str, feats_kind, dataset, f_new), index_col=0)[columns_subset].values

  mean = torch.from_numpy(mean).type(torch.float64)
  variance = torch.from_numpy(variance).type(torch.float64)
  return mean, variance

def loop(args, exp_num):
  BookKeeper._set_seed(args)
  path2data = args.path2data
  dataset = args.dataset
  lmksSubset = args.lmksSubset
  split = (args.train_frac, args.dev_frac)
  idx_dependent = args.idx_dependent
  batch_size = 1
  time = None
  seed = args.seed
  mask = args.mask
  feats_kind = args.feats_kind
  transforms = ['quat']
  f_new = args.f_new

  ## Load data iterables
  data = Data(path2data, dataset, lmksSubset, None,
              split=split, batch_size=batch_size, time=time,
              shuffle=False,
              mask=mask,
              feats_kind=feats_kind,
              f_new=f_new)

  train = data.train

  input_shapes = data.input_shape ## TODO Hardcoded
  pose_size = input_shapes['pose_size']
  trajectory_size = input_shapes['trajectory_size']
  input_shape = pose_size + trajectory_size
  
  running_sum = torch.zeros(input_shape).double()
  running_energy = torch.zeros(input_shape).double()
  running_count = 0

  ## Save files ## TODO Hardcoded
  if feats_kind == 'euler':
    columns = ['root_tx', 'root_ty', 'root_tz', 'root_rx', 'root_ry', 'root_rz'] + data.columns[1]
  elif feats_kind == 'quaternion':   ## for quaternions
    columns = ['root_tx', 'root_ty', 'root_tz', 'root_rw', 'root_rx', 'root_ry', 'root_rz'] + data.columns[1]
  elif feats_kind == 'rifke':
    columns = ['root_tx', 'root_ty', 'root_tz'] + data.columns[1]
    columns = ['root_ty'] + data.columns[1]
    
  #pre = Transforms(transforms, columns, seed, mask, feats_kind)

  for count, batch in tqdm(enumerate(train)):
    pose, trajectory, _ = batch['input']
    x = torch.cat((trajectory, pose), dim=-1)
    #x = pre.transform(x)

    x = x.squeeze(0)
    running_sum += x.sum(dim=0)
    running_energy += (x**2).sum(dim=0)
    running_count += x.shape[0]

  mean = running_sum/running_count
  energy = running_energy/running_count
  variance = energy - mean**2

  ## Add some small value to the dimensions that have zero variance to avoid nan errors
  eps = 1e-30
  zero_mask = (variance == 0)
  variance = variance + (zero_mask*eps).type(torch.float64)

  mask_str = '_'.join(['{}'.format(i) for i in mask])
  os.makedirs('dataProcessing/meanVar', exist_ok=True)
  pd.DataFrame(data=mean.reshape(1,-1).numpy(), columns=columns).to_csv('dataProcessing/meanVar/mean_{}_{}_{}_{}_{}.csv'.format(seed, mask_str, feats_kind, dataset, f_new))
  pd.DataFrame(data=variance.reshape(1,-1).numpy(), columns=columns).to_csv('dataProcessing/meanVar/variance_{}_{}_{}_{}_{}.csv'.format(seed, mask_str, feats_kind, dataset, f_new))

if __name__ == '__main__':
  argparseNloop(loop)
