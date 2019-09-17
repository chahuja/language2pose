import argparse
import os
import json
import xml.etree.cElementTree as ET
import logging

import numpy as np
import sys
sys.path.insert(0,'common')
from transforms3dbatch import *
from utils.quaternion import *

def parse_motions(path):
    xml_tree = ET.parse(path)
    xml_root = xml_tree.getroot()
    xml_motions = xml_root.findall('Motion')
    motions = []
    if len(xml_motions) > 1:
        logging.warn('more than one <Motion> tag in file "%s", only parsing the first one', path)
    motions.append(_parse_motion(xml_motions[0], path))
    return motions


def _parse_motion(xml_motion, path):
    xml_joint_order = xml_motion.find('JointOrder')
    if xml_joint_order is None:
        raise RuntimeError('<JointOrder> not found')

    joint_names = []
    joint_indexes = []
    for idx, xml_joint in enumerate(xml_joint_order.findall('Joint')):
        name = xml_joint.get('name')
        if name is None:
            raise RuntimeError('<Joint> has no name')
        joint_indexes.append(idx)
        joint_names.append(name)

    frames = {'root_pos':[], 'root_rot':[], 'joint_pos':[]}
    xml_frames = xml_motion.find('MotionFrames')
    if xml_frames is None:
        raise RuntimeError('<MotionFrames> not found')
    for xml_frame in xml_frames.findall('MotionFrame'):
        root_pos, root_rot, joint_pos = _parse_frame(xml_frame, joint_indexes)
        frames['root_pos'].append(root_pos)
        frames['root_rot'].append(root_rot)
        frames['joint_pos'].append(joint_pos)
    return joint_names, frames


def _parse_frame(xml_frame, joint_indexes):
    xml_root_pos = xml_frame.find('RootPosition')
    xml_root_rot = xml_frame.find('RootRotation')
    n_joints = len(joint_indexes)
    xml_joint_pos = xml_frame.find('JointPosition')
    if xml_joint_pos is None:
        raise RuntimeError('<JointPosition> not found')
    root_pos = _parse_list(xml_root_pos, 3)
    root_rot = _parse_list(xml_root_rot, 3)
    joint_pos = _parse_list(xml_joint_pos, n_joints, joint_indexes)

    return root_pos, root_rot, joint_pos


def _parse_list(xml_elem, length, indexes=None):
    if indexes is None:
        indexes = range(length)
    elems = [float(x) for idx, x in enumerate(xml_elem.text.rstrip().split(' ')) if idx in indexes]
    if len(elems) != length:
        raise RuntimeError('invalid number of elements')
    return elems

def mmm2csv(src):
    joint_names, mmm_dict = parse_motions(src.as_posix())[0]
    root_pos = np.array(mmm_dict['root_pos'], dtype=np.float) * 0.001 / 0.056444
    root_rot = np.array(mmm_dict['root_rot'], dtype=np.float) 
    joint_pos = np.array(mmm_dict['joint_pos'], dtype=np.float)
    
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
            
    return joints, root_pos, root_rot, values, joint_dict

def mmm2amc(src, dest):
    joints, root_pos, root_rot, values, joint_dict = mmm2csv(src)
    axesMap = {'x':'x', 'y':'y', 'z':'z'}

    root_pos = root_pos[..., [0,2,1]]
    
    ##  convert to quaternion and back by changing the axes order
    root_rot = quat2eulerbatch(qinv_np(euler2quatbatch(root_rot, 'sxyz')[...,[0, 1, 3, 2]]), 'sxyz') * 180/np.pi
    values = quat2eulerbatch(qinv_np(euler2quatbatch(values, 'sxyz')[..., [0, 1, 3, 2]]), 'sxyz') * 180/np.pi
    joint_pos = []
    
    for cnt, joint in enumerate(joints):
        for axes_num, axes in enumerate(['x', 'y', 'z']):
            if axesMap[axes] in joint_dict[joint]:
                joint_dict[joint][axesMap[axes]] = values[cnt, :, axes_num]

    lines = ["#!OML:ASF H:",
             ":FULLY-SPECIFIED",
             ":DEGREES"]
    for idx in range(root_pos.shape[0]):
        lines.append('{}'.format(idx+1))
        lines.append('root' + (' {}'*6).format(root_pos[idx, 0], root_pos[idx, 1], root_pos[idx, 2],
                                               root_rot[idx, 0], root_rot[idx, 1], root_rot[idx, 2]))
        for cnt, joint in enumerate(joint_dict):
            format_str = '{} ' * (len(joint_dict[joint])+1)
            format_str = format_str[:-1]
            joint_vals = []
            for axes in ['x', 'y', 'z']:
                if axes in joint_dict[joint]:
                    joint_vals.append(joint_dict[joint][axes][idx])
            lines.append(format_str.format(*([joint] + joint_vals)))

    lines = '\n'.join(lines) + '\n'

    os.makedirs(dest.parent, exist_ok=True)
    with open(dest, 'w') as fp:
        fp.writelines(lines)
    
