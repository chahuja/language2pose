# Modification of a version borrowed from https://github.com/facebookresearch/QuaterNet
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import matplotlib
matplotlib.use('Agg')
# from matplotlib import rc
# rc('text', usetex=True)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
import subprocess
import os
import pdb
from tqdm import tqdm
import pdb
from pathlib import Path

def plot_xzPlane(ax, minx, maxx, miny, maxy, minz):
  ## Plot a plane XZ
  verts = [
    [minx, miny, minz],
    [minx, maxy, minz],
    [maxx, maxy, minz],
    [maxx, miny, minz]
    ]
  xz_plane = Poly3DCollection([verts])
  xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
  ax.add_collection3d(xz_plane)
  return ax

def render_animation(figures, skeleton, fps, num_frames, output='out.mp4', bitrate=1000, audios=[], start_time=0, figsize=(4,4), userStudy=0, history=0, history_offset=1, suptitle=''):
    """
    Render or show an animation. The supported output modes are:
    figures:: [[kind, pos, datas, (elev, azim)]]
    kind:: graph, skeleton, highlight
    pos:: (1,2,2)
    datas:: [data1, data2 ...]

     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    initialized = [False for _ in figures]
    liness = []
    for figure in figures:
        kind, pos, datas, title, axes_label, view_point = figure
        if kind == 'skeleton':
            liness.append([[[] for _ in range(history+1)] for _ in datas])
        else:
            liness.append([])

    plt.ioff()
    fig = plt.figure(figsize=figsize)
    if userStudy:
      txt1 = """SCALE:     1    -    2    -    3    -    4    -    5    -    6    -    7
                  Disagree         Somewhat       Somewhat         Agree
                                    Disagree           Agree                 """

      txt2 = """    1. There is enough conversation to comment on the quality of the interactions of Person A.
      2. The motion of person A looks natural and match his/her speech
      3. “Person A” behaves as herself / himself (recall the reference video)
      4. “Person A” reacts realistically to person B (in terms of person B’s speech and motion)'"""
      plt.figtext(0.5, 0.2, txt1, ha='center', fontsize=16)
      plt.figtext(0.25, 0.01, txt2, ha='left', fontsize=16)
      plt.suptitle("Person A is black and red", ha='center', fontsize=16)

    if suptitle:
      plt.suptitle(suptitle, ha='center', fontsize=12)
      
    def init_func(fig, figures, skeleton, figsize, fps):
        x = 0
        y = 1
        z = 2
        radius = torch.max(skeleton.offsets()).item() * 10 # Heuristic that works well with many skeletons

        skeleton_parents = skeleton.parents()

        axes = []
        skel_fig_num = 0
        for fig_num, values in enumerate(figures):
            kind, pos, datas, title, axes_label, view_point = values
            elev, azim = view_point

            if kind == 'skeleton':
                ax = fig.add_subplot(pos[0], pos[1], pos[2], projection='3d')
                if title:
                    ax.set_title(title)
                if axes_label:
                    ax.set_xlabel(axes_label[0])
                    ax.set_ylabel(axes_label[1])
                ax.view_init(elev=elev, azim=azim)

                ax.set_xlim3d([-radius/2, radius/2])
                ax.set_ylim3d([0, radius])
                ax.set_zlim3d([0, radius])
                #ax.set_aspect('equal')

                ## plot xzplane
                MINS = datas[0].min(axis=0).min(axis=0)
                MAXES = datas[0].max(axis=0).max(axis=0)
                ax = plot_xzPlane(ax, MINS[0], MAXES[0], MINS[2], MAXES[2], 0)
                
                ax.grid(b=False)
                #ax.grid(b=True, axis='y') ## remove grid lines Hardcoded TODO, can be made a parameter
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.dist = 7.5

                trajectory = sum([data[:, 0, [0, 2]] for data in datas])/len(datas)
                avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
                draw_offset = int(25/avg_segment_length)
                spline_line, = ax.plot(*trajectory.T)
                camera_pos = trajectory
                height_offset = np.min([np.min(data[:, :, 1]) for data in datas]) # Min height

                #data = data.copy()
                for i, data in enumerate(datas):
                    figures[fig_num][2][i][:, :, 1] -= height_offset
                axes.append(ax)
                skel_fig_num = fig_num

                # min_x = np.min([np.min(data[:, :, 0]) for data in datas]) # Min height
                # max_x = np.max([np.max(data[:, :, 0]) for data in datas]) # Min height
                # min_y = np.min([np.min(data[:, :, 1]) for data in datas]) # Min height
                # max_y = np.max([np.max(data[:, :, 1]) for data in datas]) # Min height
                # min_z = np.min([np.min(data[:, :, 2]) for data in datas]) # Min height
                # max_z = np.max([np.max(data[:, :, 2]) for data in datas]) # Min height
                
                #ax.set_xlim3d([min_x, max_x])
                #ax.set_ylim3d([min_y, max_y])
                #ax.set_zlim3d([min_z, max_z])


            if kind == 'graph':
                ax = fig.add_subplot(pos[0], pos[1], pos[2])
                ax.set_xlim([0, fps])
                ax.set_ylim([np.min(datas), np.max(datas)])
                if title:
                    ax.set_title(title)
                if axes_label:
                    ax.set_xlabel(axes_label[0])
                    ax.set_ylabel(axes_label[1])
                ln = ax.plot([], [], color='red')
                liness[fig_num].append(ln[0])

                axes.append(ax)

            if kind == 'highlight':
                axes.append(axes[skel_fig_num])
                ax = axes[-1]
                ln = ax.plot([], [], [], color='g', ls='', marker='.')
                liness[fig_num].append(ln[0])

        return fig, figures, skeleton, skeleton_parents, radius, camera_pos, trajectory, draw_offset, spline_line, x, y, z, axes

    def update(frame, fig, figures, skeleton, skeleton_parents, radius, camera_pos, trajectory, draw_offset, spline_line, x, y, z, axes):
        nonlocal initialized
        nonlocal liness
        nonlocal fps
        nonlocal history
        nonlocal history_offset
        nonlocal output
        
        for fig_num, values in enumerate(zip(initialized, figures, axes, liness)):
            Initialized, (kind, pos, datas, title, axes_label, view_point), ax, lines = values
            if kind == 'skeleton':
                #orange and purple respectively['#d95f02', '#7570b3']]
                #color_list = [['red', 'black'], ['orange', 'purple']]
                color_list = [['black', 'red'], ['black', 'purple']]
                #ax = fig.add_subplot(pos[0], pos[1], pos[2], projection='3d')
                ax.set_xlim3d([-radius/2 + camera_pos[frame, 0], radius/2 + camera_pos[frame, 0]])
                ax.set_ylim3d([-radius/2 + camera_pos[frame, 1], radius/2 + camera_pos[frame, 1]])
                
                positions_world = [[data[fr] for fr in range(frame, frame-history*history_offset-1,-history_offset)] for data in datas]
                for i in range(positions_world[0][0].shape[0]):
                    if skeleton_parents[i] == -1: ## assuming 0 is body_world
                        continue
                    if not Initialized:
                        for count in range(len(datas)):
                            alpha_range = np.arange(1, 0, -1./(history+1))
                            for hist, alpha in zip(range(history+1), alpha_range):
                                col = color_list[count][0] if i in skeleton.joints_right() else color_list[count][1] # As in audio cables :)
                                #col = color_list[count][0] if hist == 0 else color_list[count][1]
                                liness[fig_num][count][hist].append(ax.plot([positions_world[count][hist][i, x], positions_world[count][hist][skeleton_parents[i], x]],
                                                                            [positions_world[count][hist][i, y], positions_world[count][hist][skeleton_parents[i], y]],
                                                                            [positions_world[count][hist][i, z], positions_world[count][hist][skeleton_parents[i], z]],
                                                                            zdir='y', c=col, alpha=alpha, marker='.'))
                    else:
                        for count in range(len(datas)):
                            for hist in range(history,-1,-1):
                                try:
                                    liness[fig_num][count][hist][i-1][0].set_xdata([positions_world[count][hist][i, x], positions_world[count][hist][skeleton_parents[i], x]])
                                except:
                                    pdb.set_trace()
                                liness[fig_num][count][hist][i-1][0].set_ydata([positions_world[count][hist][i, y], positions_world[count][hist][skeleton_parents[i], y]])
                                liness[fig_num][count][hist][i-1][0].set_3d_properties([positions_world[count][hist][i, z], positions_world[count][hist][skeleton_parents[i], z]], zdir='y')

                l = max(frame-draw_offset, 0)
                r = min(frame+draw_offset, trajectory.shape[0])
                spline_line.set_xdata(trajectory[l:r, 0])
                spline_line.set_ydata(np.zeros_like(trajectory[l:r, 0]))
                spline_line.set_3d_properties(trajectory[l:r, 1], zdir='y')
                initialized[fig_num] = True

            if kind == 'graph':
                if frame <= fps/2:
                    ax.set_xlim([0, fps])
                    liness[fig_num][0].set_data(range(0, frame), datas[0:frame])                
                else:
                    ax.set_xlim([frame-fps/2, frame+fps/2])
                    liness[fig_num][0].set_data(range(int(frame-fps/2), frame), datas[int(frame-fps/2):frame])

            if kind == 'highlight':
                outputs, mask = datas
                inv_mask = 1-mask
                ln = liness[fig_num][0]
                non_zero_indices = [idx for idx in range(outputs[0].shape[1]) if not mask[frame, idx] == 0]
                ln.set_xdata(outputs[0][frame, non_zero_indices, x])
                ln.set_ydata(outputs[0][frame, non_zero_indices, y])
                ln.set_3d_properties(outputs[0][0, non_zero_indices, z], zdir='y')

        ## save as images as well
        #filename = '.'.join(Path(output).name.split('.')[:-1])
        #out_folder = '.'.join(output.split('.')[:-1])
        #os.makedirs(out_folder, exist_ok=True) 
        #plt.savefig(out_folder + '/{}_{:05d}.png'.format(filename, frame))
        
            
    fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=np.arange(history*history_offset, num_frames), interval=1000/fps, repeat=False, fargs=init_func(fig, figures, skeleton, figsize, fps))
    if output == 'interactive':
        plt.show()
        return anim
    elif output == 'html':
        return anim.to_html5_video()
    elif output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)

        if audios:
            temp_output = output + '_temp.mp4'
            anim.save(temp_output, writer=writer)

            ## add audio to the output
            command_inputs = []
            for audio in reversed(audios):
                command_inputs.append('-ss')
                command_inputs.append('{}'.format(start_time))
                command_inputs.append('-i')
                command_inputs.append('{}'.format(audio))

            command = ['ffmpeg', '-y']
            command += command_inputs
            command.append('-i')
            command.append('{}'.format(temp_output))
            command.append('-filter_complex')
            command.append('[0:a][1:a]amerge=inputs=2[aout]')
            command.append('-map')
            command.append('[aout]')
            command.append('-map')
            command.append('0:a')
            command.append('-map')
            command.append('1:a')
            command.append('-map')
            command.append('2:v')
            command.append('-acodec')
            command.append('ac3')
            command.append('-shortest')
            command.append('{}'.format(output))
            FNULL = open(os.devnull, 'w')
            subprocess.call(command, stderr=FNULL, stdout=FNULL)

            delete_command = ['rm', '{}'.format(temp_output)]
            subprocess.call(delete_command, stderr=FNULL, stdout=FNULL)
        else:
            anim.save(output, writer=writer)

        
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only html, .mp4, and .gif are supported)')
    plt.close()


def plot_trajectories():
    pass
