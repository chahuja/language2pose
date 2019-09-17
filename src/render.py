import torch
import torch.nn as nn

from dataUtils import *
from lossUtils import *
from model.model import *
from data.data import *
from common.transforms3dbatch import *
from common.parallel import parallel
from data.kit_visualization import render, render4
from renderUtils import *

from pycasper.name import Name
from pycasper.BookKeeper import *
from argsUtils import argparseNloop

import numpy as np
from tqdm import tqdm
import pdb
import os

def loop(args, exp_num):
  # assert args.load, 'Model name not provided'
  # assert os.path.isfile(args.load), 'Model file not found'

  args_subset = ['exp', 'cpk', 'model', 'time', 'chunks']
  book = BookKeeper(args, args_subset, args_dict_update={'feats_kind':args.feats_kind,
                                                         'render_list':args.render_list})
  args = book.args

  if args.load:
    dir_name = book.name.dir(args.save_dir)
  else:
    dir_name = args.path2data

  if args.render_list is not None:
    with open(args.render_list, 'r') as f:
      render_list = f.readlines()
      render_list = {filename.strip() for filename in render_list}
  else:
    render_list = None

  dataset = args.dataset
  feats_kind = args.feats_kind
  if dataset == 'KITMocap':
    path2data = '../dataset/kit-mocap'
  elif dataset == 'CMUMocap':
    #path2data = '../dataset/cmu-pose/all_asfamc'
    raise NotImplementedError

  data = Data(path2data, dataset, lmksSubset=['all'], desc=None, load_data=False)

  ## Load Skeleton
  skel = pkl.load(open('dataProcessing/{}/skeleton.p'.format(args.dataset), 'rb'))
  filenames = []
  descriptions = []
  outputs = []

  feats_kind_dict = {'quaternion':'csv',
                     'fke':'fke',
                     'rifke':'rifke'}

  idx = 1
  for tup in os.walk(dir_name):
    for filename in tup[2]:
      if filename.split('.')[-1] == feats_kind_dict[feats_kind] and Path(tup[0]).name != 'new':
        if render_list:  ## only render the files in render list
          if filename.split('_')[0] not in render_list:
            continue
        output = Path(tup[0])/'videos'/filename
        if not args.clean_render:    ## only render files which do not exist. Useful if rendering was interrupted/incomplete 
          if output.with_suffix('.mp4').exists():
            continue
        outputs.append(output.with_suffix('.mp4').as_posix())
        descriptions.append(get_description(data.df, filename, path2data, feats_kind))
        os.makedirs(output.parent, exist_ok=True)
        filename = Path(tup[0])/filename
        filenames.append(filename.as_posix())

  print('{} files'.format(len(filenames)))

  parallelRender(filenames, descriptions, outputs, skel, args.feats_kind)
  
if __name__ == '__main__':
  argparseNloop(loop)
