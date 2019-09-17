import os

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

# import sys
# sys.path.insert(0, './data')
# sys.path.insert(0, './utils')
# sys.path.insert(0, './common')
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from utils.visualization import *
from utils.skeleton import Skeleton
from common.mmm import parse_motions
from common.transforms3dbatch import *
from common.quaternion import *
from renderUtils import quat2xyz
from model.model import Integrator

import torch
import pickle as pkl
import scipy.ndimage.filters as filters

import pdb


## permute joints to make it a DAG
def permute(parents, root=0, new_parent=-1, new_joints=[], new_parents=[]):
  new_joints.append(root)
  new_parents.append(new_parent)
  new_parent = len(new_joints) - 1
  for idx, p in enumerate(parents):
    if p == root:
      permute(parents, root=idx, new_parent=new_parent, new_joints=new_joints, new_parents=new_parents)
  return new_joints, new_parents

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)

class RawData():
  def __init__(self):
    pass

  def _get_f(self):
    raise NotImplementedError
  
  def _get_df(self):
    raise NotImplementedError

  def preProcess(self):
    raise NotImplementedError
  
  def get_skeletonNpermutation(self):
    raise NotImplementedError
  
  @property
  def quat_columns(self):
    ## quaternion columns
    quat_columns = ['root_tx', 'root_ty', 'root_tz']
    for joint in self.skel.joints:
      quat_columns += ['{}_{}'.format(joint, col_suffix) for col_suffix in ['rw', 'rx', 'ry', 'rz']]

    return quat_columns

  @property
  def fke_columns(self):
    ## forward kinematics columns
    fke_columns = []
    for joint in self.skel.joints:
      fke_columns += ['{}_{}'.format(joint, col_suffix) for col_suffix in ['tx', 'ty', 'tz']]

    return fke_columns

  @property
  def pose_columns(self):
    pose_columns = []
    for joint in self.skel.joints:
      pose_columns += ['{}_{}'.format(joint, col_suffix) for col_suffix in ['rx', 'ry', 'rz']]

    return pose_columns
  
  @property
  def rifke_columns(self):
    ## Save Rotation invariant fke (rifke)
    rifke_columns = self.fke_columns + ['root_Vx', 'root_Vz', 'root_Ry', 'feet_l1', 'feet_l2', 'feet_r1', 'feet_r2']
    return rifke_columns

  @property
  def rifke_dict(self):
    raise NotImplementedError

  def output_columns(self, feats_kind):
    if feats_kind in {'euler'}:
      return self.pose_columns
    elif feats_kind in {'quaternion'}:
      return self.quat_columns
    elif feats_kind in {'fke'}:
      return self.fke_columns
    elif feats_kind in {'rifke'}:
      return self.rifke_columns

  def mat2csv(self, data, filename, columns):
    pd.DataFrame(data=data, columns=columns).to_csv(filename)
  
  def quat2fke(self, df_quat, filename_fke, filename_rifke):
    '''Save Forward Kinematics'''
    df_fke = pd.DataFrame(data=np.zeros((df_quat.shape[0], len(self.fke_columns))), columns=self.fke_columns)
    ## copying translation as is
    df_fke[['root_tx', 'root_ty', 'root_tz']] = df_quat.loc[:, ['root_tx', 'root_ty', 'root_tz']].copy()
    xyz_data = quat2xyz(df_quat, self.skel)
    df_fke.loc[:, self.fke_columns] = xyz_data.reshape(-1, np.prod(xyz_data.shape[1:]))
    #filename_fke = dir_name / Path(row[feats_kind]).relative_to(Path(path2data)/'subjects').with_suffix('.fke')
    os.makedirs(filename_fke.parent, exist_ok=True)
    df_fke.to_csv(filename_fke.as_posix())

    '''Save Rotation Invariant Forward Kinematics'''
    df_rifke = pd.DataFrame(data=np.zeros((df_quat.shape[0]-1, len(self.rifke_columns))), columns=self.rifke_columns)
    rifke_data = self.fke2rifke(xyz_data.copy())
    df_rifke[self.rifke_columns] = rifke_data[..., 3:]
    #filename_rifke = dir_name / Path(row[feats_kind]).relative_to(Path(path2data)/'subjects').with_suffix('.rifke')
    os.makedirs(filename_rifke.parent, exist_ok=True)
    df_rifke.to_csv(filename_rifke.as_posix())

    ''' Convert rifke to fke to get comparable ground truths '''
    new_df_fke = pd.DataFrame(data=self.rifke2fke(df_rifke[self.rifke_columns].values, filename_rifke).reshape(-1, len(self.fke_columns)),
                              columns=self.fke_columns)
    new_fke_dir = filename_fke.parent/'new_fke' 
    os.makedirs(new_fke_dir, exist_ok=True)
    new_df_fke.to_csv((new_fke_dir/filename_fke.name).as_posix())

    return xyz_data

  ## fke to rotation invariant fke (Holden et. al.)
  def fke2rifke(self, positions):
    """ Put on Floor """
    #fid_l, fid_r = np.array([5,6]), np.array([10,11])
    fid_l, fid_r = self.rifke_dict['fid_l'], self.rifke_dict['fid_r']
    foot_heights = np.minimum(positions[:,fid_l,1], positions[:,fid_r,1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    
    positions[:,:,1] -= floor_height

    """ Add Reference Joint """
    trajectory_filterwidth = 3
    reference = positions[:,0] * np.array([1,0,1])
    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')    
    positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)
    
    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.05,0.05]), np.array([3.0, 2.0])
    
    feet_l_x = (positions[1:,fid_l,0] - positions[:-1,fid_l,0])**2
    feet_l_y = (positions[1:,fid_l,1] - positions[:-1,fid_l,1])**2
    feet_l_z = (positions[1:,fid_l,2] - positions[:-1,fid_l,2])**2
    feet_l_h = positions[:-1,fid_l,1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
    
    feet_r_x = (positions[1:,fid_r,0] - positions[:-1,fid_r,0])**2
    feet_r_y = (positions[1:,fid_r,1] - positions[:-1,fid_r,1])**2
    feet_r_z = (positions[1:,fid_r,2] - positions[:-1,fid_r,2])**2
    feet_r_h = positions[:-1,fid_r,1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    
    """ Get Root Velocity """
    velocity = (positions[1:,0:1] - positions[:-1,0:1]).copy()
    
    """ Remove Translation """
    positions[:,:,0] = positions[:,:,0] - positions[:,0:1,0]
    positions[:,:,2] = positions[:,:,2] - positions[:,0:1,2]
    
    """ Get Forward Direction """
    #sdr_l, sdr_r, hip_l, hip_r = 19, 26, 3, 8
    sdr_l, sdr_r, hip_l, hip_r = self.rifke_dict['sdr_l'], self.rifke_dict['sdr_r'], self.rifke_dict['hip_l'], self.rifke_dict['hip_r']
    
    across1 = positions[:,hip_l] - positions[:,hip_r]
    across0 = positions[:,sdr_l] - positions[:,sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    rotation = qbetween_np(forward, target)[:, np.newaxis]
    positions = qrot_np(np.repeat(rotation, positions.shape[1], axis=1), positions)
    
    """ Get Root Rotation """
    velocity = qrot_np(rotation[1:], np.repeat(velocity, rotation.shape[1], axis=1))
    rvelocity = self.get_rvelocity(rotation, forward='z', plane='xz')
    
    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
    positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)
    return positions

  def get_rvelocity(self, rotation, forward='z', plane='xz'):
    ## TODO - might need a reversal of inputs for qmul_np
    qs = qmul_np(rotation[1:], qinv_np(rotation[:-1]))
    ds = np.zeros(qs.shape[:-1] + (3,))
    ds[...,'xyz'.index(forward)] = 1.0
    ds = qrot_np(qs, ds)
    ys = ds[...,'xyz'.index(plane[0])]
    xs = ds[...,'xyz'.index(plane[1])]
    return np.arctan2(ys, xs)

  def rifke2fke(self, positions, filename=None):
    root_ry = torch.from_numpy(positions[..., -5]).unsqueeze(0).unsqueeze(0).float()
    pos = positions[..., :-7].reshape(positions.shape[0], -1, 3)
    pos[..., 0, [0,2]] = 0

    ''' Get Y Rotations '''
    integrator = Integrator(1, root_ry.shape[-1])
    root_ry = integrator(root_ry).squeeze(0).squeeze(0).numpy()
    rotations = np.stack([np.cos(root_ry/2), np.zeros_like(root_ry),
                          np.sin(root_ry/2), np.zeros_like(root_ry)],
                         axis=-1).astype(np.float)
    rotations = np.expand_dims(rotations, axis=1)

    ''' Rotate positions by adding Y rotations '''
    pos = qrot_np(np.repeat(qinv_np(rotations), pos.shape[1], axis=1), pos)

    ''' Rotate XZ velocity vector '''
    root_v = positions[..., -7:-5]
    root_v = np.stack([root_v[..., 0], np.zeros_like(root_v[..., 0]), root_v[..., 1]], axis=-1)
    try:
      root_v = qrot_np(qinv_np(rotations.squeeze(1)), root_v)
    except:
      pdb.set_trace()
    root_v = torch.from_numpy(root_v.transpose(1,0)).unsqueeze(0).float()

    ''' Get Root Positions from Root Velocities'''
    integrator = Integrator(3, root_v.shape[-1])
    root_t = integrator(root_v).squeeze(0).transpose(1, 0).numpy()

    ''' Add translations back to all the joints '''
    pos[..., :, 0] += root_t[..., 0:1]
    pos[..., :, 2] += root_t[..., 2:3]
    
    return pos

class KITMocap(RawData):
  def __init__(self, path2data, preProcess_flag=False):
    super(KITMocap, self).__init__()
    ## load skeleton
    self._SKELPATH = 'dataProcessing/KITMocap/skeleton.p'
    self._MMMSKELPATH = 'skeleton/mmm.xml'
    self._MMMSAMPLEPATH = 'dataProcessing/KITMocap/00001_mmm.xml'
    os.makedirs(Path(self._SKELPATH).parent, exist_ok=True)
    ## get the skeleton and permutation
    self.skel, self.permutation, self.new_joints = self.get_skeletonNpermutation()

    ## save skeleton
    pkl.dump(self.skel, open(self._SKELPATH, 'wb'))

    if preProcess_flag:
      self.preProcess(path2data)

    ## Reading data
    data = []
    for tup in os.walk(path2data):
      for filename in tup[2]:
        if Path(filename).suffix == '.xml':
          annotpath = Path(tup[0])/(filename.split('_')[0] + '_annotations.json')
          annot = json.load(open(annotpath, 'r'))
          quatpath = filename.split('_')[0] + '_quat.csv'
          fkepath = filename.split('_')[0] + '_quat.fke'
          rifkepath = filename.split('_')[0] + '_quat.rifke'
          if annot:
            for description in annot:
              data.append([(Path(tup[0])/filename).as_posix(),
                           description,
                           (Path(tup[0])/quatpath).as_posix(),
                           (Path(tup[0])/fkepath).as_posix(),
                           (Path(tup[0])/rifkepath).as_posix()])
          else:
            data.append([(Path(tup[0])/filename).as_posix(),
                         '',
                         (Path(tup[0])/quatpath).as_posix(),
                         (Path(tup[0])/fkepath).as_posix(),
                         (Path(tup[0])/rifkepath).as_posix()])
              
    self.df = pd.DataFrame(data=data, columns=['euler', 'descriptions', 'quaternion', 'fke', 'rifke'])

    self.columns = pd.read_csv(self.df.iloc[0].quaternion, index_col=0).columns
    joints = [col[:-3] for col in self.columns]
    self.joints = []
    self.columns_dict = {}
    start = 0
    for joint in joints:
      if not self.joints:
        self.joints.append(joint)
        end = 1
      elif self.joints[-1] == joint:
        end += 1
      else:
        self.columns_dict.update({self.joints[-1]:self.columns[start:end]})
        self.joints.append(joint)
        start = end
        end = end + 1
    self.columns_dict.update({self.joints[-1]:self.columns[start:end]})

  def _get_df(self):
    return self.df

  def _get_f(self):
    return 100

  @property
  def rifke_dict(self):
    return {'fid_l':np.array([14,15]),
            'fid_r':np.array([19,20]),
            'sdr_l':6,
            'sdr_r':9,
            'hip_l':12,
            'hip_r':17}
  
  def preProcess(self, path2data):
    print('Preprocessing KIT Data')
    for tup in os.walk(path2data):
      for filename in tqdm(tup[2]):
        if Path(filename).suffix == '.xml':
          filepath = Path(tup[0])/filename
          quatpath = filename.split('_')[0] + '_quat.csv'
          quatpath = (Path(tup[0])/quatpath).as_posix()
          xyz_data, skel, joints, root_pos, rotations = self.mmm2quat(filepath)
          ## create quat dataframe
          root_pos = root_pos.squeeze(0)
          rotations = rotations.contiguous().view(rotations.shape[1], -1)
          quats = torch.cat([root_pos, rotations], dim=-1).numpy()
          columns = ['root_tx', 'root_ty', 'root_tz'] + \
                    ['{}_{}'.format(joint, axis) for joint in joints for axis in ['rw', 'rx', 'ry', 'rz']]
          df = pd.DataFrame(data=quats, columns=columns)
          df.to_csv(quatpath)
          filename_fke = Path(quatpath).with_suffix('.fke')
          filename_rifke = Path(quatpath).with_suffix('.rifke')
          self.quat2fke(df, filename_fke, filename_rifke)

  def mat2amc(self, data, filename):
    lines = ["#!OML:ASF H:",
             ":FULLY-SPECIFIED",
             ":DEGREES"]
    for count, row in enumerate(data):
      start = 0
      lines.append('{}'.format(count+1))
      for joint in self.joints:
        end = start + len(self.columns_dict[joint])
        format_str = '{} '* (len(self.columns_dict[joint]) + 1)
        format_str = format_str[:-1] ## remove the extra space
        lines.append(format_str.format(*([joint] + list(row[start:end]))))
        start = end
    lines = '\n'.join(lines) + '\n'

    os.makedirs(filename.parent, exist_ok=True)
    with open(filename,'w') as fp:
      fp.writelines(lines)

  def get_new_parents(self, parents, joints_left, joints_right, joints):
    permutation, new_parents = permute(parents)
    joints_w_root = ['root'] + joints
    new_joints = [joints_w_root[perm] for perm in permutation]
    new_joints_idx = list(range(len(new_joints)))
    new_joints_left = []
    new_joints_right = []
    for idx, jnt in enumerate(new_joints):
      if jnt[0] == 'R':
        new_joints_right.append(idx)
      else:
        new_joints_left.append(idx)
        
    return permutation, new_parents, new_joints_left, new_joints_right, new_joints

  ## KITMocap Specific
  def get_skeletonNpermutation(self):
    ## make a parents_list
    parents = [-1, 3, 0, 2, 1, 8, 9, 0, 7, 1, 6, 12, 5, 16, 17, 0, 15, 1, 14, 20, 13]
    joints_left = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    joints_right = [13, 14, 15, 16, 17, 18, 19, 20]

    ## read a demo file to get the joints
    joints, _, _, _ = self.mmm2csv(Path(self._MMMSAMPLEPATH))
    
    permutation, new_parents, new_joints_left, new_joints_right, new_joints = self.get_new_parents(parents, joints_left, joints_right, joints)

    import xml.etree.ElementTree as ET
    tree = ET.parse(self._MMMSKELPATH)
    root = tree.getroot()

    ## make an offset list
    offset_dict = self.get_offsets(root, joints)
    offset_dict.update({'root':[0,0,0]})

    new_offsets = [offset_dict[joint] for joint in new_joints]

    ## make a Skeleton
    skel = Skeleton(new_offsets, new_parents, new_joints_left, new_joints_right, new_joints)
    return skel, permutation, new_joints
    
  ## read an xml file
  def mmm2csv(self, src):
    joint_names, mmm_dict = parse_motions(src.as_posix())[0]
    root_pos = np.array(mmm_dict['root_pos'], dtype=np.float) #* 0.001 / 0.056444
    #root_pos = root_pos[:, [1,2,0]]
    root_rot = np.array(mmm_dict['root_rot'], dtype=np.float) #* 180/np.pi
    #root_rot = root_rot[:, [1,2,0]]
    joint_pos = np.array(mmm_dict['joint_pos'], dtype=np.float) #* 180/np.pi

    joint_dict = {}
    for idx, name in enumerate(joint_names):
      if name.split('_')[0][-1] != 't':
        xyz = name.split('_')[0][-1]
        joint = name.split('_')[0][:-1]
      else:
        xyz = 'y'
        joint = name.split('_')[0]
      if joint not in joint_dict:
        joint_dict[joint] = dict()
      joint_dict[joint][xyz] = joint_pos[:, idx]

    joints = []
    values = []
    for cnt, joint in enumerate(joint_dict):
      joint_vals = []
      joints.append(joint)
      for axes in ['x', 'y', 'z']:
        if axes in joint_dict[joint]:
          joint_vals.append(joint_dict[joint][axes])
        else:
          joint_vals.append(np.zeros_like(root_pos[:, 0]))
      values.append(np.stack(joint_vals, axis=1))
    values = np.stack(values, axis=0)

    return joints, root_pos, root_rot, values

  def get_offsets(self, root, Joints):
    joints = root.findall('RobotNode')
    offset_dict = {}
    for joint in joints:
      matrix = joint.findall('Transform')
      if matrix:
        offset = []
        ## switch y and z axis
        for row in ['row1', 'row3', 'row2']:
          Row = matrix[0].findall('Matrix4x4')[0].findall(row)
          offset.append(float(Row[0].attrib['c4']))
        joint_name = joint.attrib['name']
        if joint_name.split('_')[0][-6:] == 'egment':
          if joint_name[:-13] in Joints:
            offset_dict[joint_name[:-13]] = offset
        else:
          if joint_name[:-6] in Joints:
            offset_dict[joint_name[:-6]] = offset
          elif joint_name[:-7] in Joints:
            offset_dict[joint_name[:-7]] = offset
    return offset_dict

  def mmm2quat(self, path):
    joints, root_pos, root_rot, values = self.mmm2csv(path)

    ## convert to quaternions
    values_quat = euler2quatbatch(values, axes='sxyz')
    root_rot_quat = euler2quatbatch(root_rot, axes='sxyz')

    ## switch y and z axis
    ## Note the qinv_np is very important as 2 axes are being interchanged - can be proved using basic vector equations
    root_pos = root_pos[..., [0, 2, 1]] 
    values_quat = qinv_np(values_quat[..., [0, 1, 3, 2]])
    root_rot_quat = qinv_np(root_rot_quat[..., [0, 1, 3, 2]])

    rotations = np.expand_dims(np.transpose(np.concatenate((np.expand_dims(root_rot_quat, axis=0), values_quat), axis=0), axes=[1, 0, 2]), axis=0)
    root_pos = np.expand_dims(root_pos, axis=0)

    new_rotations = torch.from_numpy(rotations[:, :, self.permutation, :])
    new_root_pos = torch.from_numpy(root_pos.copy())

    xyz_data = self.skel.forward_kinematics(new_rotations, new_root_pos)[0]
    return xyz_data.numpy(), self.skel, self.new_joints, new_root_pos, new_rotations

class CMUMocap(RawData):
  def __init__(self, path2data, preProcess_flag=False):
    super(CMUMocap, self).__init__()
    ## load skeleton
    self._SKELPATH = 'dataProcessing/CMUMocap/skeleton.p'
    self._MMMSKELPATH = 'dataProcessing/CMUMocap/01.asf'
    self._MMMSAMPLEPATH = None
    os.makedirs(Path(self._SKELPATH).parent, exist_ok=True)
    ## get the skeleton and permutation
    self.skel, self.permutation, self.new_joints = self.get_skeletonNpermutation()
    ## save skeleton
    pkl.dump(self.skel, open(self._SKELPATH, 'wb'))

    if preProcess_flag:
      self.preProcess(path2data)

    ## Reading the Descriptions df
    self.desc_df = self.read_desc(path2data/Path('descriptions'))

    ## Reading the feats df and adding descriptions along the way
    data = []
    for tup in os.walk(path2data/Path('subjects')):
      for filename in tup[2]:
        if Path(filename).suffix == '.csv':
          descriptions = self.desc_df[self.desc_df['file'] == Path(filename).stem]['descriptions']
          if descriptions.shape[0]: ## add the data only if the description files are available
            quatpath = Path(tup[0]).parent.parent/'subjects_quaternion'/Path(tup[0]).name/filename
            fkepath = quatpath.with_suffix('.fke')
            rifkepath = quatpath.with_suffix('.rifke')
            data.append([(Path(tup[0])/filename).as_posix(),
                         descriptions.iloc[0],
                         quatpath.as_posix(),
                         fkepath.as_posix(),
                         rifkepath.as_posix()])


    self.df = pd.DataFrame(data=data, columns=['euler', 'descriptions', 'quaternion', 'fke', 'rifke'])

    ## figuring out self.columns/columns_dict/joints
    self.columns = pd.read_csv(self.df.iloc[0].euler, index_col=0).columns
    joints = [col[:-3] for col in self.columns]
    self.joints = []
    self.columns_dict = {}
    start = 0
    for joint in joints:
      if not self.joints:
        self.joints.append(joint)
        end = 1
      elif self.joints[-1] == joint:
        end += 1
      else:
        self.columns_dict.update({self.joints[-1]:self.columns[start:end]})
        self.joints.append(joint)
        start = end
        end = end + 1
    self.columns_dict.update({self.joints[-1]:self.columns[start:end]})

  def _get_df(self):
    return self.df

  def _get_f(self):
    return 120

  @property
  def rifke_dict(self):
    return {'fid_l':np.array([5,6]),
            'fid_r':np.array([10,11]),
            'sdr_l':19,
            'sdr_r':26,
            'hip_l':3,
            'hip_r':8}

  def preProcess(self, path2data):
    print('Preprocessing CMUMocap data')
    ## Read Skeleton
    columns, joints, columns_dict, offset_dict, axis_dict, _, _, _ = self.read_skeleton(open(self._MMMSKELPATH))
    self.columns = columns
    self.columns_dict = columns_dict
    ## Read motion files
    for tup in tqdm(os.walk(Path(path2data)/'subjects')):
      for filename in tup[2]:
        if Path(filename).suffix == '.amc':
          fp_pose = open(Path(tup[0])/filename)
          df, columns_ordered, joints_ordered = self.read_pose(fp_pose, columns, joints, columns_dict, axis_dict)
          outputPath = (Path(tup[0])/Path(filename)).with_suffix('.csv').as_posix()
          df.to_csv(outputPath)

          ## get quaternions
          filename = Path(path2data) / 'subjects_quaternion' / (Path(tup[0])/filename).relative_to(Path(path2data)/'subjects').with_suffix('.csv')
          df_quat = self.euler2quat(df, axis_dict, filename)

          ## get fke and rifke
          filename_fke = filename.with_suffix('.fke')
          filename_rifke = filename.with_suffix('.rifke')
          self.quat2fke(df_quat, filename_fke, filename_rifke)

  def get_new_parents(self, parents, joints_left, joints_right, joints):
    permutation, new_parents = permute(parents)
    joints_w_root = joints
    new_joints = [joints_w_root[perm] for perm in permutation]
    new_joints_idx = list(range(len(new_joints)))
    new_joints_left = []
    new_joints_right = []
    for idx, jnt in enumerate(new_joints):
      if jnt[0] == 'r':
        new_joints_right.append(idx)
      else:
        new_joints_left.append(idx)
        
    return permutation, new_parents, new_joints_left, new_joints_right, new_joints

  ## CMUMocap Specific
  def get_skeletonNpermutation(self):
    '''make a parents_list'''
    ## read a demo file to get the joints
    columns, joints, columns_dict, offset_dict, axis_dict, parents, joints_left, joints_right = self.read_skeleton(open(self._MMMSKELPATH))
    #parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 21, 20, 13, 24, 25, 26, 27, 28, 27]
    #parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 11, 15, 16, 17, 18, 19, 18, 11, 22, 23, 24, 25, 26, 25]
    #joints_left = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    #joints_right = [5, 6, 7, 8, 22, 23, 24, 25, 26, 27, 28]

    permutation, new_parents, new_joints_left, new_joints_right, new_joints = self.get_new_parents(parents, joints_left, joints_right, joints)

    ## make an offset list
    offset_dict.update({'root':[0,0,0]})
    new_offsets = [offset_dict[joint] for joint in new_joints]
    
    ## make a Skeleton
    skel = Skeleton(new_offsets, new_parents, new_joints_left, new_joints_right, new_joints)
    return skel, permutation, new_joints
    
  def read_skeleton(self, fp):
    '''lhipjoint and rhipoint do not have any degrees of freedom, hence they are ignored'''
    columns = []
    columns_dict = {}
    joints = []
    offset_dict = {}
    axis_dict = {}
    get_cols = lambda joint, dof: ['{}_{}'.format(joint, DOF) for DOF in dof] 
    ## first read the root
    for line in fp:
      if line.strip() == ':root':
        joint = 'root'
        dof = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        columns_ = get_cols(joint, dof)
        columns += columns_
        columns_dict.update({joint:columns_})
        joints.append(joint)
      elif line.strip().split(' ')[0] == 'name':
        joint = line.strip().split(' ')[-1]
        joints.append(joint)
      elif line.strip().split(' ')[0] == 'dof':
        dof = line.strip().split(' ')[1:]
        columns_ = get_cols(joint, dof)
        columns += columns_
        columns_dict.update({joint:columns_})
      elif line.strip().split(' ')[0] == 'direction':
        direction = np.array(line.strip().split(' ')[1:], dtype=np.float)
      elif line.strip().split(' ')[0] == 'length':
        length = float(line.strip().split(' ')[-1])
        try:
          offset_dict[joint] = list(direction * length)
        except:
          continue
      elif line.strip().split(' ')[0] == 'axis':
        if joint == 'root':
          axis_dict.update({joint:list(np.array([0, 0, 0], dtype=np.float64))})
        else:
          axis = list(np.array(line.strip().split(' ')[1:4], dtype=np.float64))
          axis_dict.update({joint:axis})
      elif line.strip() == ':hierarchy':
        break
    
    ## Build tree
    parents = np.zeros((len(joints)), dtype=np.int)
    parents[0] = -1 ## assuming the first joint is always the root
    joints_rev = {jnt:i for i, jnt in enumerate(joints)}
    joints_left = [i for i, jnt in enumerate(joints) if jnt[0]!='r']
    joints_right = [i for i, jnt in enumerate(joints) if jnt[0]=='r']
    for line in fp:
      if line.strip() == 'begin':
        continue
      elif line.strip() == 'end':
        break
      else:
        node = line.strip().split(' ')
        p = node[0]
        c = node[1:]
        for child in c:
          parents[joints_rev[child]] = joints_rev[p]
    parents = list(parents)
    return columns, joints, columns_dict, offset_dict, axis_dict, parents, joints_left, joints_right

  def read_pose(self, fp, columns, joints, columns_dict, axis_dict):
    offset = 3 ## lines of meta-data on top
    columns_ordered = []
    joints_ordered = []
    pose = []
    pose_partial = []
    missing_columns = []
    missing_joints = []
    
    ## Find missing columns and joints
    for joint in joints:
      if joint not in columns_dict:
        missing_columns += ['{}_{}'.format(joint, dof) for dof in ['rx', 'ry', 'rz']]
        missing_joints += [joint]
    for j, line in enumerate(fp):
      line = line.strip().split(' ')
      if line[0] in columns_dict:
        if j >= offset and j <= (offset+1+len(columns_dict)):
          columns_ordered += columns_dict[line[0]]
          joints_ordered.append(line[0])
        pose_partial += line[1:]
        if len(pose_partial) == len(columns):
          ## add missing columns
          pose_partial += [0] * len(missing_columns)
          ## add the completed frame to the pose list
          pose.append(pose_partial)
          pose_partial = []
          
    ## add missing columns and joints to columns_ordered and joints_ordered
    columns_ordered += missing_columns
    joints_ordered += missing_joints

    data = np.array(pose, dtype=np.float64)
    return pd.DataFrame(data=data, columns=columns_ordered), columns_ordered, joints_ordered

  ## Euler to AMC
  def mat2amc(self, data, filename):
    lines = ["#!OML:ASF H:",
             ":FULLY-SPECIFIED",
             ":DEGREES"]
    for count, row in enumerate(data):
      start = 0
      lines.append('{}'.format(count+1))
      for joint in self.joints:
        if joint in self.columns_dict:
          end = start + len(self.columns_dict[joint])
          format_str = '{} '* (len(self.columns_dict[joint]) + 1)
          format_str = format_str[:-1] ## remove the extra space
          lines.append(format_str.format(*([joint] + list(row[start:end]))))
          start = end
    lines = '\n'.join(lines) + '\n'

    os.makedirs(filename.parent, exist_ok=True)
    with open(filename,'w') as fp:
      fp.writelines(lines)

  def euler2quat(self, df, axis_dict, filename):
    ## find the missing_columns and joints
    columns = df.columns
    joints = []
    missing_columns = []
    for column in columns:
      col = column[:-3]
      if col not in joints:
        joints.append(col)
        for col_suffix in ['rx', 'ry', 'rz']:
          column_ = '{}_{}'.format(col, col_suffix)
          if column_ not in columns:
            missing_columns.append(column_)
      else:
          pass

    self.joints = joints
    ## axes to C
    axes = []
    for joint in self.skel.joints:
      axes.append(axis_dict[joint])
    axes = np.deg2rad(np.array(axes, dtype=np.float64))
    C = euler2matbatch(axes)
    C_inv = np.linalg.inv(C)

    '''
    child_matrix = P_coordinate + P_matrix * (C*R*C_inv * offset) 
    where,
    R = euler2matbatch(euler)
    C = euler2matbatch(axes)
    C_inv = C^{-1}
    offset = length * distance

    q = mat2quatbatch(C*R*C_inv)

    To get the euler values back,
    euler = mat2eulerbatch(C_inv * quat2matbatch(q) * C)
    '''
    df_missing_cols = pd.DataFrame(data=np.zeros((df.shape[0], len(missing_columns))), columns=missing_columns)
    df = pd.concat([df, df_missing_cols], axis=1)

    df_quat = pd.DataFrame(data=np.zeros((df.shape[0], len(self.quat_columns))), columns=self.quat_columns)
    ## copying translation as is
    df_quat[['root_tx', 'root_ty', 'root_tz']] = df.loc[:, ['root_tx', 'root_ty', 'root_tz']].copy()

    euler = df[self.pose_columns].values
    euler = euler.reshape(euler.shape[0], -1, 3) * np.pi/180
    R = euler2matbatch(euler, axes='sxyz')
    euler_cap = np.matmul(np.matmul(C, R), C_inv)
    quaternion = mat2quatbatch(euler_cap)

    euler_cap_ = quat2matbatch(quaternion)
    R_ = np.matmul(np.matmul(C_inv, euler_cap_), C)
    euler_ = mat2eulerbatch(R_, axes='sxyz')

    euler_ = euler_.reshape(euler.shape[0], int(euler.shape[1]*3)) * 180/np.pi
    quaternion = quaternion.reshape(euler.shape[0], int(euler.shape[1]*4))

    ### save the euler conversion
    df[self.pose_columns] = euler_
    euler_filename = Path('test.amc')
    self.mat2amc(df[columns].values, euler_filename)

    ## Save Quaternions
    df_quat[self.quat_columns[3:]] = quaternion
    filename = filename.with_suffix('.csv')
    os.makedirs(filename.parent, exist_ok=True)
    df_quat.to_csv(filename.as_posix())
    return df_quat
    
  '''
  Read the description files
  '''
  def read_desc(self, directory):
    df = pd.DataFrame(columns=['file', 'descriptions'])
    for tup in os.walk(directory):
      for filename in tup[2]:
        if Path(filename).suffix == '.csv':
          filename = Path(tup[0])/filename
          with open(filename, 'r') as f:
            lines = f.readlines()
          for count, line in enumerate(lines):
            if count >=3:
              line = line.strip().split('\t')
              if line[0] != '':
                df = df.append({'file':line[0], 'descriptions':line[-1]}, ignore_index=True)
    return df

if __name__ == '__main__':
  """PreProcessing"""
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-dataset', default='CMUMocap', type=str,
                      help='dataset kind')
  parser.add_argument('-path2data', default='../dataset/cmu-pose/all_asfamc/', type=str,
                      help='dataset kind')
  args, _ = parser.parse_known_args()
  eval(args.dataset)(args.path2data, preProcess_flag=True)
  print('Succesfully Preprocessed {} data'.format(args.dataset))
