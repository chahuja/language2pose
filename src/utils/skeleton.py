# Borrowed from https://github.com/facebookresearch/QuaterNet

import torch
import numpy as np
from utils.quaternion import qmul_np, qmul, qrot

class Skeleton:
    def __init__(self, offsets, parents, joints_left=None, joints_right=None, joints=None):
        assert len(offsets) == len(parents)
        
        self._offsets = torch.FloatTensor(offsets)
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._joints = joints
        self._compute_metadata()
    
    def cuda(self):
        self._offsets = self._offsets.cuda()
        return self
    
    def num_joints(self):
        return self._offsets.shape[0]
    
    def offsets(self):
        return self._offsets
    
    def parents(self):
        return self._parents
    
    def has_children(self):
        return self._has_children
    
    def children(self):
        return self._children
    
    def remove_joints(self, joints_to_remove, dataset):
        """
        Remove the joints specified in 'joints_to_remove', both from the
        skeleton definition and from the dataset (which is modified in place).
        The rotations of removed joints are propagated along the kinematic chain.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)
                
        # Update all transformations in the dataset
        for count, rotations in enumerate(dataset):
            for joint in joints_to_remove:
                for child in self._children[joint]:
                    rotations[:, :, child] = qmul(rotations[:, :, joint], rotations[:, :, child])
                rotations[:, :, joint] = torch.DoubleTensor([1, 0, 0, 0]) # Identity
            dataset[count] = rotations[:, :, valid_joints]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        new_joints_right = []
        new_joints_left = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
                if i in self._joints_left:
                    new_joints_left.append(len(new_parents))
                elif i in self._joints_right:
                    new_joints_right.append(len(new_parents))
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)
        self._joints_left = np.array(new_joints_left)
        self._joints_right = np.array(new_joints_right)

        self._offsets = self._offsets[valid_joints]
        self._compute_metadata()
        
    def forward_kinematics(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(rotations.shape[0], rotations.shape[1],
                                                   self._offsets.shape[0], self._offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[self._parents[i]], expanded_offsets[:, :, i]) \
                                       + positions_world[self._parents[i]])
                if self._has_children[i]:
                    rotations_world.append(qmul(rotations_world[self._parents[i]], rotations[:, :, i]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)
    
    def joints_left(self):
        return self._joints_left
    
    def joints_right(self):
        return self._joints_right

    @property
    def joints(self):
        return self._joints
        
    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)
