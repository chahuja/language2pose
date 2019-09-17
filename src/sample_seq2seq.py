import torch
import torch.nn as nn

from dataUtils import *
from lossUtils import *
from model.model import *
from data.data import *
from common.transforms3dbatch import *

from pycasper.name import Name
from pycasper.BookKeeper import *
from argsUtils import argparseNloop

import numpy as np
from tqdm import tqdm
import pdb

def get_mask_indices(mask):
  indices = []
  for i, m in enumerate(mask):
    if m:
      indices.append(i)
  return indices

## get delta based on mask
def local2global(outputs_list, start_trajectory_list, input_shape, trajectory_size, mask):
  ## use only the first start_trajectory
  outputs = torch.cat(outputs_list, dim=0)
  outputs = outputs.view(-1, input_shape)

  start_trajectory = start_trajectory_list[0][0] ## get only the first time-step
  
  mask = np.array(mask)
  indices = get_mask_indices(mask)
  start_trajectory = start_trajectory[indices]
  for t in range(outputs.shape[0]):
    outputs[t, indices] += start_trajectory
    start_trajectory = outputs[t, indices]

  return outputs

def toEuler(M, joints, euler_columns):
  columns = ['root_tx', 'root_ty', 'root_tz'] + ['{}_{}'.format(joint, suffix) for joint in joints for suffix in ['rx', 'ry', 'rz']]
  M = M.values
  quats = M[:, 3:].reshape(M.shape[0], -1, 4)
  quats = quats/((quats**2).sum(axis=-1, keepdims=True)**0.5) ## normalize the quaternions
  euler = quat2eulerbatch(quats, axes='sxyz').reshape(quats.shape[0], int(quats.shape[1]*3)) * 180/np.pi
  euler = np.concatenate([M[:, :3], euler], axis=-1)
  df = pd.DataFrame(data=euler, columns=columns)
  return df[euler_columns]
  
def toFKE(M, data, filename):
  ''' Convert RIFKE to FKE '''
  output_columns = data.raw_data.output_columns('rifke')
  M = data.raw_data.rifke2fke(M[output_columns].values)

  ''' Save FKE '''
  M = M.reshape(M.shape[0], -1)
  output_columns = data.raw_data.output_columns('fke')
  pd.DataFrame(data=M,
               columns=output_columns).to_csv(filename.as_posix())  

def sample(args, exp_num, data=None):
  assert args.load, 'Model name not provided'
  assert os.path.isfile(args.load), 'Model file not found'

  args_subset = ['exp', 'cpk', 'model', 'time', 'chunks']
  book = BookKeeper(args, args_subset, args_dict_update={})
  args = book.args

  dir_name = book.name.dir(args.save_dir)
  
  ## Training parameters
  path2data = args.path2data
  dataset = args.dataset
  lmksSubset = args.lmksSubset
  desc = args.desc
  split = (args.train_frac, args.dev_frac)
  idx_dependent = args.idx_dependent
  f_new = args.f_new
  
  ## hardcoded for sampling
  batch_size = args.batch_size
  time = args.time
  offset = args.offset
  ## mask for delta
  mask = args.mask
  
  global feats_kind
  feats_kind = args.feats_kind
  
  ## Load data iterables
  if data is None:
    data = Data(path2data, dataset, lmksSubset, desc,
                split, batch_size=batch_size,
                time=time,
                offset=offset,
                shuffle=False,
                mask=mask,
                feats_kind=feats_kind,
                f_new=f_new)

  train = data.train.dataset.datasets
  dev = data.dev.dataset.datasets
  test = data.test.dataset.datasets

  print('Data Loaded')
  
  ## Create a model
  device = torch.device('cuda:{}'.format(args.cuda)) if args.cuda>=0 else torch.device('cpu')
  input_shape = data.input_shape
  modelKwargs = {}
  modelKwargs.update(input_shape)
  modelKwargs.update(args.modelKwargs)

  model = eval(args.model)(**modelKwargs)
  model.to(device).double()

  print('Model Created')

  ## Load model
  if args.load:
    print('Loading Model')
    book._load_model(model)

  ## Loss function
  criterion = Loss(args.losses, args.lossKwargs)
  
  ## Transforms
  global columns
  columns = get_columns(feats_kind, data)
  pre = Transforms(args.transforms, columns, args.seed, mask, feats_kind, dataset, f_new)
  
  def loop(model, data, dataLoaders, pre, batch_size, desc='train'):
    running_loss = 0
    count = 0
    model.eval()

    Tqdm = tqdm(dataLoaders, desc=desc+' {:.4f}'.format(running_loss/(count+1.)), leave=False, ncols=20)
    for count, loader in enumerate(Tqdm):
      loader = DataLoader(loader, batch_size=batch_size, shuffle=False)
      outputs_list = []
      start_trajectory_list = []
      for _, batch in enumerate(loader):
        model.zero_grad()

        X, Y = batch['input'], batch['output']
        pose, trajectory, start_trajectory = X
        pose_gt, trajectory_gt, start_trajectory_gt = Y
        
        x = torch.cat((trajectory, pose), dim=-1)
        y = torch.cat((trajectory_gt, pose_gt), dim=-1)

        x = x.to(device)
        y = y.to(device)
        start_trajectory_gt = start_trajectory_gt.to(device)
        
        ## Transform before the model
        x = pre.transform(x)
        y = pre.transform(y)

        if offset == 0:
          y_cap, internal_losses = model(x, train=False)
          #y_cap = x
          #internal_losses = []
        else:
          assert 0, 'offset = {}, it must be 0 for now'.format(offset)

        input_shape = sum([data.input_shape[key] for key in data.input_shape]) 
        trajectory_size = data.input_shape['trajectory_size']

        #y_output = y_cap.repeat(1,1,loader.dataset.f_ratio).view(new_size)
        #mask = torch.Tensor(data.mask[:3]).to(y_cap.device).double().view(1,1,-1)
        #y_output[..., :3] = (y_output[..., :3] * mask) * 1./loader.dataset.f_ratio + y_output[..., :3] * (1-mask)
        outputs_list.append(pre.inv_transform(y_cap))
        start_trajectory_list.append(start_trajectory_gt)
        
        loss = criterion(y_cap, y)
        for i_loss in internal_losses:
          loss += i_loss

        running_loss += loss.item()
        ## update tqdm
        Tqdm.set_description(desc+' {:.4f}'.format(running_loss/(count+1.)))
        Tqdm.refresh()

        x = x.detach()
        y = y.detach()
        loss = loss.detach()
        y_cap = y_cap.detach()

      if outputs_list:
        ## Collect all outputs
        outputs = local2global(outputs_list, start_trajectory_list, input_shape, trajectory_size, data.mask)
        new_size = list(outputs.shape)
        new_size[0] *= loader.dataset.f_ratio
        outputs = outputs.repeat(1,loader.dataset.f_ratio).view(new_size)

        outputs = outputs.detach().cpu().numpy()

        ## copy outputs in the dataframe format
        mat_full_temp = pd.DataFrame(data=np.zeros((outputs.shape[0], len(columns))), columns=loader.dataset.mat_full.columns)

        ## copy all joints
        mat_full_temp.loc[:, columns] = outputs

        if dataset == 'KITMocap':
          filename = Path(dir_name)/Path(desc)/Path(loader.dataset.path2csv).relative_to(path2data).with_suffix('.csv')
          os.makedirs(filename.parent, exist_ok=True)
          if feats_kind == 'quaternion':
            data.raw_data.mat2csv(mat_full_temp.values, filename, columns)
          elif feats_kind == 'rifke':
            toFKE(mat_full_temp,
                  data,
                  filename.with_suffix('.fke'))

        elif dataset == 'CMUMocap':
          filename = Path(dir_name)/Path(desc)/Path(loader.dataset.path2csv).relative_to(path2data).with_suffix('.amc')
          os.makedirs(filename.parent, exist_ok=True)
          if feats_kind in {'quaternion'}:
            data.raw_data.mat2csv(mat_full_temp.values, filename, columns)
            mat_full = toEuler(M=mat_full_temp,
                               joints=data.raw_data.joints,
                               euler_columns=data.raw_data.columns)
            data.raw_data.mat2amc(mat_full.values, filename)
          elif feats_kind == 'rifke':
            toFKE(mat_full_temp,
                  data,
                  filename.with_suffix('.fke'))

      if count>=0 and args.debug: ## debugging by overfitting
        break
      
    return running_loss/(count+1.)

  ## Sample
  train_loss = loop(model, data, train, pre, batch_size, 'train')
  dev_loss = loop(model, data, dev, pre, batch_size, 'dev')
  test_loss = loop(model, data, test, pre, batch_size, 'test')

  ## update results but not save them (just to print)
  book.update_res({'train':train_loss,
                   'dev':dev_loss,
                   'test':test_loss})
  
  ## print results
  book.print_res(0, key_order=['train','dev','test'], exp=exp_num)

if __name__ == '__main__':
  argparseNloop(sample)
