import argparse
import itertools
from ast import literal_eval

def argparseNloop(loop):
  parser = argparse.ArgumentParser()

  ## Dataset Parameters
  parser.add_argument('-dataset', nargs='+', type=str, default=['CMUMocap'],
                      help='name of the dataset')
  parser.add_argument('-path2data', nargs='+', type=str, default=['../dataset/cmu-pose/all_asfamc/'],
                      help='path to data')
  parser.add_argument('-train_frac', nargs='+', type=float, default=[0.6],
                      help='Fraction of data to be used for training')
  parser.add_argument('-dev_frac', nargs='+', type=float, default=[0.2],
                      help='Fraction of data to be used as dev')
  parser.add_argument('-idx_dependent', nargs='+', type=int, default=[1],
                      help='is training person dependent?')
  parser.add_argument('-batch_size', nargs='+', type=int, default=[100],
                      help='minibatch size. Use batch_size=1 when using time=0')
  parser.add_argument('-time', nargs='+', type=int, default=[32],
                      help='time steps. time=0 gives the full sequence. Use with batch_size=1')
  parser.add_argument('-chunks', nargs='+', type=int, default=[1],
                      help='chunks of time steps. to be used with language inputs')
  parser.add_argument('-offset', nargs='+', type=int, default=[0],
                      help='offset == 0; autoencoder, offset >= 1; prediction')
  parser.add_argument('-mask', nargs='+', type=literal_eval, default=[[1, 0, 1, 0, 0, 0, 0]],
                      help='delta mask for translation as a list. if 1, use delta values')  
  parser.add_argument('-pose_mask', nargs='+', type=literal_eval, default=[0],
                      help='mask for the pose. use 1 for pose velocity for all joints. 0 for absolute values')  
  parser.add_argument('-feats_kind', nargs='+', type=str, default=['quaternion'],
                      help='feature kind; quaternion or euler')  
  parser.add_argument('-seedLength', nargs='+', type=int, default=[20],
                      help='initial length of inputs to seed the prediction; used when offset > 0')  
  parser.add_argument('-lmksSubset', nargs='+', type=literal_eval, default=[['all']],
                      help='choose from a subset of landmarks based on parts of the face like lips, eyes etc. None for the whole face.')
  parser.add_argument('-desc', nargs='+', type=literal_eval, default=[None],
                      help='choose a subset of the dataset based on the given description')
  parser.add_argument('-s2v', nargs='+', type=int, default=[0],
                      help='choose a subset of the dataset based on the given description')
  parser.add_argument('-transforms', nargs='+', type=literal_eval, default=[['zNorm']],
                      help='choose from a set of multiple transforms like normalization and pca')
  parser.add_argument('-f_new', nargs='+', type=int, default=[0],
                      help='subsample to the new frequency, use 0 for using original frequency')  
  parser.add_argument('-view', nargs='+', type=str, default=['sentences.txt'],
                      help='list of sentences to sample from')  
  
  parser.add_argument('-exp', nargs='+', type=int, default=[None],
                      help='experiment number')
  parser.add_argument('-debug', nargs='+', type=int, default=[0],
                      help='debug mode')
  parser.add_argument('-save_dir', nargs='+', type=str, default=['save/model'],
                      help='directory to store checkpointed models')
  parser.add_argument('-cpk', nargs='+', type=str, default=['m'],
                      help='checkpointed model name')
  parser.add_argument('-tb', nargs='+', type=int, default=[0],
                      help='Tensorboard Flag')
  parser.add_argument('-seed', nargs='+', type=int, default=[11212],
                      help='manual seed')
  parser.add_argument('-load', nargs='+', type=str, default=[None],
                      help='Load weights from this file')
  parser.add_argument('-cuda', nargs='+', type=int, default=[0],
                      help='choice of gpu device, -1 for cpu')
  parser.add_argument('-overfit', nargs='+', type=int, default=[0],
                      help='disables early stopping and saves models even if the dev loss increases. useful for performing an overfitting check')

  ## model hyperparameters
  parser.add_argument('-model', nargs='+', type=str, default=['Autoencoder'],
                      help='choice of model')
  parser.add_argument('-modelKwargs', nargs='+', type=literal_eval, default=[{'num_channels_list':[40, 40, 20]}],
                      help='choice of model arguments')

  ## Loss hyperparameters
  parser.add_argument('-losses', nargs='+', type=literal_eval, default=[['MSELoss']],
                      help='choice of losses MSELoss, SmoothL1loss etc.')
  parser.add_argument('-lossKwargs', nargs='+', type=literal_eval, default=[[{'reduction':'sum'}]],
                      help='kwargs corresposing to the losses')

  ## training parameters
  parser.add_argument('-num_epochs', nargs='+', type=int, default=[50],
                      help='number of epochs for training')
  parser.add_argument('-early_stopping', nargs='+', type=int, default=[1],
                      help='Use 1 for early stopping')
  parser.add_argument('-greedy_save', nargs='+', type=int, default=[1],
                      help='save weights after each epoch if 1')
  parser.add_argument('-save_model', nargs='+', type=int, default=[1],
                      help='flag to save model at every step')
  parser.add_argument('-stop_thresh', nargs='+', type=int, default=[3],
                      help='number of consequetive validation loss increses before stopping')
  parser.add_argument('-eps', nargs='+', type=float, default=[0],
                      help='if the decrease in validation is less than eps, it counts for one step in stop_thresh ')

  ## Training Parameters
  parser.add_argument('-curriculum', nargs='+', type=int, default=[0],
                      help='learn generating time steps by starting with 2 timesteps upto time, increasing by a power of 2')
  parser.add_argument('-kl_anneal', nargs='+', type=int, default=[0],
                      help='anneal kl loss till the number of epochs')
  
  ## optimization paramters
  parser.add_argument('-lr', nargs='+', type=float, default=[0.001],
                      help='learning rate')

  ## dataProcessing/augmentDataset.py parameters
  parser.add_argument('-angles', nargs='+', type=literal_eval, default=[[90]],
                      help='set of angles to augment data. Example: [-90, 90]')  
  
  ## slurm_generator Parameters
  parser.add_argument('-config', nargs='+', type=str, default=[None],
                      help='Config file to generate slurm job files')
  parser.add_argument('-script', nargs='+', type=str, default=[None],
                      help='script to use for the job files')


  ## render.py Params
  parser.add_argument('-clean_render', nargs='+', type=int, default=[1],
                      help='render all videos from scratch if True')  
  parser.add_argument('-render_list', nargs='+', type=str, default=[None],
                      help='render videos only from the render list')

  ## render_after_training.py Params
  parser.add_argument('-render', nargs='+', type=str, default=['inf'],
                      help='samples, new or inf')  
  
    
  args, unknown = parser.parse_known_args()
  print(args)
  print(unknown)

  ## Create a permutation of all the values in argparse
  args_dict = args.__dict__
  args_keys = sorted(args_dict)
  args_perm = [dict(zip(args_keys, prod)) for prod in itertools.product(*(args_dict[names] for names in args_keys))]

  for i, perm in enumerate(args_perm):
    args.__dict__.update(perm)
    print(args)
    loop(args, i)
