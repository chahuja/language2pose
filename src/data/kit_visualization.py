''' Depricated '''
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from utils.visualization import *
from utils.skeleton import Skeleton
from common.mmm import parse_motions
from common.transforms3dbatch import *
from utils.visualization import render_animation
from utils.quaternion import qinv_np

## read an xml file
def mmm2csv(src):
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

def get_offsets(root, Joints):
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


def mmm2quat(path):
  pdb.set_trace()
  joints, root_pos, root_rot, values = mmm2csv(path)

  ## convert to quaternions
  values_quat = euler2quatbatch(values, axes='sxyz')
  root_rot_quat = euler2quatbatch(root_rot, axes='sxyz')

  ## switch y and z axis
  root_pos = root_pos[..., [0, 2, 1]] 
  values_quat = qinv_np(values_quat[..., [0, 1, 3, 2]])
  root_rot_quat = qinv_np(root_rot_quat[..., [0, 1, 3, 2]])
  
  ## make a parents_list
  parents = [-1, 3, 0, 2, 1, 8, 9, 0, 7, 1, 6, 12, 5, 16, 17, 0, 15, 1, 14, 20, 13]
  joints_left = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  joints_right = [13, 14, 15, 16, 17, 18, 19, 20]

  ## permute joints to make it a DAG
  def permute(parents, root=0, new_parent=-1, new_joints=[], new_parents=[]):
    new_joints.append(root)
    new_parents.append(new_parent)
    new_parent = len(new_joints) - 1
    for idx, p in enumerate(parents):
      if p == root:
        permute(parents, root=idx, new_parent=new_parent, new_joints=new_joints, new_parents=new_parents)
    return new_joints, new_parents

  permutation, new_parents = permute(parents)
  joints_w_root = ['root'] + joints
  new_joints = [joints_w_root[perm] for perm in permutation]
  new_joints_idx = list(range(len(new_joints)))
  #new_left_joints = sorted([new_joints_idx[perm] for idx, perm in enumerate(permutation) if idx in joints_left])
  new_joints_left = []
  new_joints_right = []
  for idx, jnt in enumerate(new_joints):
    if jnt[0] == 'R':
      new_joints_right.append(idx)
    else:
      new_joints_left.append(idx)

  import xml.etree.ElementTree as ET
  tree = ET.parse('skeleton/mmm.xml')
  root = tree.getroot()

  ## make an offset list
  offset_dict = get_offsets(root, joints)
  offset_dict.update({'root':[0,0,0]})

  new_offsets = [offset_dict[joint] for joint in new_joints]

  ## make a Skeleton
  skel = Skeleton(new_offsets, new_parents, new_joints_left, new_joints_right)
  rotations = np.expand_dims(np.transpose(np.concatenate((np.expand_dims(root_rot_quat, axis=0), values_quat), axis=0), axes=[1, 0, 2]), axis=0)
  root_pos = np.expand_dims(root_pos, axis=0)

  pdb.set_trace()
  new_rotations = torch.from_numpy(rotations[:, :, permutation, :])
  new_root_pos = torch.from_numpy(root_pos.copy())

  xyz_data = skel.forward_kinematics(new_rotations, new_root_pos)[0]
  return xyz_data.numpy(), skel, new_joints, new_root_pos, new_rotations

def render4(xyz_data, skel, time, output, figsize):
  ## render animation
  render_animation([['skeleton', (2,2,2), [xyz_data], None, None, [90, 0]],
                    ['skeleton', (2,2,1), [xyz_data], None, None, [0, 0]],
                    ['skeleton', (2,2,4), [xyz_data], None, None, [0, 90]],
                    ['skeleton', (2,2,3), [xyz_data], None, None, [30, 20]]],
                   skel, 
                   100,
                   min(time, int(xyz_data.shape[0])),
                   output=output,
                   figsize=figsize)

def render(xyz_data, skel, time, output, figsize, description):
  ## render animation
  render_animation([['skeleton', (1,1,1), [xyz_data], None, None, [30, -65]]],
                   skel, 
                   100,
                   min(time, int(xyz_data.shape[0])), 
                   output=output,
                   figsize=figsize,
                   suptitle=description)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-out', type=str, default='out.mp4',
                      help='filename of the output')
  parser.add_argument('-time', type=int, default=500,
                      help='length of output')

  args = parser.parse_args()

  path = Path('../dataset/kit-mocap/03112_mmm.xml')
  xyz_data, skel, _, _, _ = mmm2quat(path)
  render(xyz_data,
         skel,
         min(args.time, int(xyz_data.shape[0])),
         args.out,
         (4, 4))
