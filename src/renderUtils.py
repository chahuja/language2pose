import torch
import pandas as pd
from data.kit_visualization import render, render4
from common.parallel import parallel
from argsUtils import argparseNloop

import numpy as np
import pdb
from pathlib import Path


def fill_missing_joints(df, joints):
  joints_in_df = set([col[:-3] for col in df.columns])
  missing_joints = set(joints) - joints_in_df
  missing_xyz = ['{}_{}'.format(jnt, xyz) for jnt in missing_joints for xyz in ['rx','ry','rz']]
  missing_w = ['{}_rw'.format(jnt) for jnt in missing_joints]

  df_missing_xyz = pd.DataFrame(data=np.zeros((df.shape[0], len(missing_xyz))), columns=missing_xyz)
  df_missing_w = pd.DataFrame(data=np.ones((df.shape[0], len(missing_w))), columns=missing_w)  

  return pd.concat([df, df_missing_w, df_missing_xyz], axis=1)

def quat2xyz(df, skel):
  df = fill_missing_joints(df, skel.joints)
  root_pos = torch.from_numpy(df[['root_tx', 'root_ty', 'root_tz']].values).unsqueeze(0)
  columns = [str_format.format(joint) for joint in skel.joints for str_format in ['{}_rw', '{}_rx', '{}_ry', '{}_rz']]
  root_orientation = torch.from_numpy(df[columns].values)
  root_orientation = root_orientation.view(1, root_orientation.shape[0], -1, 4)
  xyz_data = skel.forward_kinematics(root_orientation, root_pos)[0].numpy()
  return xyz_data

def readNrender(params):
  filenum, filename, description, skel, time, output, figsize, feats_kind = params
  df = pd.read_csv(filename, index_col=0)
  if feats_kind == 'quaternion':
    xyz_data = quat2xyz(df, skel)
  elif feats_kind == 'fke':
    xyz_data = df.values.reshape(df.shape[0], -1, 3)

  render(xyz_data, skel, time, output, figsize, description)
  print(filenum, filename)


def chunk_description(description, max_len=40):
  description = description.split(' ')
  chunks = []
  chunk = ''
  length = 0
  for desc in description:
    length += len(desc) + 1
    if length > max_len:
      chunks.append(chunk)
      length = len(desc) + 1
      chunk = desc + ' '
    else:
      chunk += desc + ' '
  if chunk:
    chunks.append(chunk)
  description = '\n'.join(chunks)
  return description

def get_description(df, filename, path2data, feats_kind):
  filename = (Path(path2data)/filename).as_posix()
  description = df[df[feats_kind] == filename].iloc[0]['descriptions']
  description = chunk_description(description)
  return description

def parallelRender(filenames, descriptions, outputs, skel, feats_kind):
  filenums = [i for i in range(len(filenames))]
  skels = [skel for _ in range(len(filenames))]
  times = [np.inf for _ in range(len(filenames))]
  figsizes = [(4,4) for _ in range(len(filenames))]
  feats_kind = [feats_kind] * len(filenames)

  #for input in zip(filenums, filenames, descriptions, skels, times, outputs, figsizes, feats_kind):
    #readNrender(input)
  parallel(readNrender, zip(filenums, filenames, descriptions, skels, times, outputs, figsizes, feats_kind))

def renderOnSlurm(dir_name, dataset, feats_kind):
  pass
