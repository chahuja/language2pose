import torch
import torch.nn as nn

from dataUtils import *
from lossUtils import *
from model.model import *
from data.data import *
from common.transforms3dbatch import *
from renderUtils import parallelRender

from pycasper.name import Name
from pycasper.BookKeeper import *
from argsUtils import argparseNloop
from slurmpy import Slurm

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
  book = BookKeeper(args, args_subset, args_dict_update={'view':args.view})
  args = book.args

  dir_name = book.name.dir(args.save_dir)
  
  ## Training parameters
  path2data = args.path2data
  dataset = args.dataset
  lmksSubset = args.lmksSubset
  desc = args.desc
  split = (args.train_frac, args.dev_frac)
  idx_dependent = args.idx_dependent

  ## hardcoded for sampling
  batch_size = args.batch_size
  time = args.time
  chunks = args.chunks
  offset = args.offset
  ## mask for delta
  mask = args.mask
  
  global feats_kind
  feats_kind = args.feats_kind
  global render_feats_kind
  render_feats_kind = {'rifke':'fke',
                       'quaternion':'quaternion',
                       'fke':'fke'}
  s2v = args.s2v
  f_new = args.f_new
  curriculum = args.curriculum
  
  ## Load data iterables
  if data is None:
    data = Data(path2data, dataset, lmksSubset, desc,
                split, batch_size=batch_size,
                time=time,
                chunks=chunks,
                offset=offset,
                shuffle=False,
                mask=mask,
                feats_kind=feats_kind,
                s2v=s2v,
                f_new=f_new)
    print('Data Loaded')
  else:
    print('Data already loaded!! Yessss!')
        
  train = data.train.dataset.datasets
  dev = data.dev.dataset.datasets
  test = data.test.dataset.datasets

  
  ## Create a model
  global device
  device = torch.device('cuda:{}'.format(args.cuda)) if args.cuda>=0 else torch.device('cpu')
  input_shape = data.input_shape
  modelKwargs = {}
  modelKwargs.update(input_shape)
  modelKwargs.update(args.modelKwargs)

  ## getting the input_size
  if args.s2v:
    input_size = 300
  elif args.desc:
    input_size = len(args.desc)
  else:
    input_size = 0

  model = eval(args.model)(chunks, input_size=input_size, Seq2SeqKwargs=modelKwargs, load=None)
  model.to(device).double()

  print('Model Created')

  ## Load model
  if args.load:
    print('Loading Model')
    book._load_model(model)

  ## Transforms
  global columns
  columns = get_columns(feats_kind, data)
  pre = Transforms(args.transforms, columns, args.seed, mask, feats_kind, dataset, f_new)
  
  def loop(model, data, loader, sentences, pre, batch_size, desc='train'):
    filenames = []
    output_filenames = []
    descriptions = sentences
    model.eval()
    for count, s2v in enumerate(tqdm(sentences)):
      try:
        row = data.df[data.df['descriptions'] == s2v].iloc[0]
        minidata = MiniData(row[feats_kind], args.lmksSubset, args.time,
                            sentence_vector=row['descriptions'],
                            mask=args.mask,
                            dataset=args.dataset,
                            f_ratio=int(data.raw_data._get_f()/args.f_new),
                            feats_kind=feats_kind)
        start = minidata[0]['input']
        start = np.concatenate([start[1], start[0]], axis=-1)
        start = pre.transform(torch.from_numpy(start).to(device).double())[0, 0:1]
      except:
        start = torch.zeros(1, data.input_shape['pose_size'] + data.input_shape['trajectory_size']).to(device).double()
      y_cap, internal_losses = model.sample([s2v], time_steps=32, start=start)
      #outputs_list.append(pre.inv_transform(y_cap))
      #outputs = torch.cat(outputs_list, dim=0)
      outputs = pre.inv_transform(y_cap).squeeze(0)
      new_size = list(outputs.shape)
      new_size[0] *= loader[0].f_ratio
      outputs = outputs.repeat(1,loader[0].f_ratio).view(new_size)
      outputs = outputs.detach().cpu().numpy()
      ## copy outputs in the dataframe format
      mat_full_temp = pd.DataFrame(data=np.zeros((outputs.shape[0], len(columns))), columns=columns)

      if feats_kind == 'rifke':
        mat_full_temp['root_tx'] = 0
        mat_full_temp['root_tz'] = 0

      ## copy all joints
      mat_full_temp.loc[:, columns] = outputs
      if dataset == 'KITMocap':
        filename = Path(dir_name)/Path('new')/Path('{}.csv'.format(count))
        filenames.append(filename.with_suffix('.fke').as_posix())
        output_filenames.append(filename.with_suffix('.mp4').as_posix())
        os.makedirs(filename.parent, exist_ok=True)
        if feats_kind == 'quaternion':
          data.raw_data.mat2csv(mat_full_temp.values, filename, columns)
        elif feats_kind == 'rifke':
          toFKE(mat_full_temp,
                data,
                filename.with_suffix('.fke'))

    ## Render
    #parallelRender(filenames, descriptions, output_filenames, data.raw_data.skel, render_feats_kind[feats_kind])
    ## Render on slurm (cpu-node only)
    # print('Rendering')
    # render = Slurm('render_new', slurm_kwargs={'partition':'cpu_long', 'time':'10-00:00', 'n':10})
    # python_cmd = ['source activate torch',
    #               'python render_newSentence.py -load {} -view {}'.format(
    #                 args.load,
    #                 args.view)]
    # render.run('\n'.join(python_cmd))

  ## Sample
  with open(args.view, 'r') as f:
    sentences = f.readlines()
  sentences = [s.strip() for s in sentences]
  loop(model, data, train, sentences, pre, batch_size, 'train')

if __name__ == '__main__':
  argparseNloop(sample)
