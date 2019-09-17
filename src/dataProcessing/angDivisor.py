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

def loadAngDivisor(columns_subset, seed):
  angDivisor = pd.read_csv('dataProcessing/angDivisor_{}.csv'.format(seed), index_col=0)[columns_subset].as_matrix()
  angDivisor = torch.from_numpy(angDivisor).type(torch.float64)
  return angDivisor

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

  ## Load data iterables
  data = Data(path2data, dataset, lmksSubset, None,
              split=split, batch_size=batch_size, time=time,
              shuffle=False)

  columns = ['root_tx', 'root_ty', 'root_tz', 'root_rx', 'root_ry', 'root_rz'] + data.columns[1]
  angDivisor = torch.ones(len(columns))
  angDivisor[3:] = angDivisor[3:]*360
  pd.DataFrame(data=angDivisor.reshape(1,-1).numpy(), columns=columns).to_csv('dataProcessing/angDivisor_{}.csv'.format(seed))


if __name__ == '__main__':
  argparseNloop(loop)
