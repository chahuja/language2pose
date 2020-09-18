## to run dataparallel - s2v must be a tensor
## sampling can be made more efficient
## slurm pipelining - train->sample->render
## better render in pygame
## do not load seq2seq model for training the best model
## but we need it as a baseline

## sentence_dec_cross is same accross all input sentences. If that is true, then the sentence embedding is learning nothing about the pose
## maybe learn the input embedding using an LSTM instead of BERT
## OR maybe train for the pose output first and then learn to predict sentences
## OR maybe do some cyclic loss training

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from dataUtils import *
from lossUtils import *
from model.model import *
from data.data import *

from pycasper.name import Name
from pycasper.BookKeeper import *
from argsUtils import argparseNloop
from sample_wordConditioned import *
from sample_wordConditioned_newSentence import sample as render_new_sentences
from slurmpy import Slurm

import numpy as np
from tqdm import tqdm
import copy

def train(args, exp_num, data=None):
  #assert args.load, 'Model name not provided'
  #assert os.path.isfile(args.load), 'Model file not found'
  if args.load and os.path.isfile(args.load):
    load_pretrained_model=True
  else:
    load_pretrained_model=False
  args_subset = ['exp', 'cpk', 'model', 'time', 'chunks']
  book = BookKeeper(args, args_subset, args_dict_update={'chunks':args.chunks,
                                                         'batch_size':args.batch_size,
                                                         'model':args.model,
                                                         's2v':args.s2v,
                                                         'cuda':args.cuda,
                                                         'save_dir':args.save_dir,
                                                         'early_stopping':args.early_stopping,
                                                         'debug':args.debug,
                                                         'stop_thresh':args.stop_thresh,
                                                         'desc':args.desc,
                                                         'curriculum':args.curriculum,
                                                         'lr':args.lr},
                    tensorboard=args.tb,
                    load_pretrained_model=load_pretrained_model)
  ## load_pretrained_model makes sure that the model is loaded, old save files are not updated and _new_exp is called to assign new filename

  args = book.args

  ## Start Log
  book._start_log()
  
  ## Training parameters
  path2data = args.path2data
  dataset = args.dataset
  lmksSubset = args.lmksSubset
  desc = args.desc
  split = (args.train_frac, args.dev_frac)
  idx_dependent = args.idx_dependent
  batch_size = args.batch_size
  time = args.time
  global chunks
  chunks = args.chunks
  offset = args.offset
  mask = args.mask
  feats_kind = args.feats_kind
  s2v = args.s2v
  f_new = args.f_new
  curriculum = args.curriculum

  if args.debug:
    shuffle=False
  else:
    shuffle=True
  
  ## Load data iterables
  if data is None:
    data = Data(path2data, dataset, lmksSubset, desc,
                split, batch_size=batch_size,
                time=time,
                chunks=chunks,
                offset=offset,
                shuffle=shuffle,
                mask=mask,
                feats_kind=feats_kind,
                s2v=s2v,
                f_new=f_new)
    print('Data Loaded')
  else:
    print('Data already loaded! Yesss!!')

  train = data.train
  dev = data.dev
  test = data.test

  
  ## Create a model
  device = torch.device('cuda:{}'.format(args.cuda)) if args.cuda>=0 else torch.device('cpu')
  input_shape = data.input_shape
  kwargs_keys = ['pose_size', 'trajectory_size']
  modelKwargs = {key:input_shape[key] for key in kwargs_keys}
  modelKwargs.update(args.modelKwargs)

  ## TODO input_size is hardcoded to the w2v input size. can be extracted from Data
  if args.s2v:
    input_size = 300
  elif args.desc:
    input_size = len(args.desc)
  else:
    input_size = 0
  model = eval(args.model)(chunks, input_size=input_size, Seq2SeqKwargs=modelKwargs, load=args.load)
  model.to(device).double()
  book._copy_best_model(model)
  print('Model Created')

  ## would have to skip this way of loading model
  #if args.load:
  #  print('Loading Model')
  #  book._load_model(model)

  ## Loss function
  criterion = Loss(args.losses, args.lossKwargs)

  ## Optimizers
  optim = torch.optim.Adam(model.parameters(), lr=args.lr)

  ## LR scheduler
  scheduler = lr_scheduler.ExponentialLR(optim, gamma=0.99)
  
  ## Transforms
  columns = get_columns(feats_kind, data)
  pre = Transforms(args.transforms, columns, args.seed, mask, feats_kind, dataset, f_new)

  def loop(model, data, pre, desc='train', epoch=0):
    running_loss = 0
    running_internal_loss = 0
    running_count = 0
    
    if desc == 'train':
      model.train(True)
    else:
      model.eval()

    Tqdm = tqdm(data, desc=desc+' {:.4f}'.format(0), leave=False, ncols=20)
    for count, batch in enumerate(Tqdm):
      model.zero_grad()
      optim.zero_grad()
      X, Y, s2v = batch['input'], batch['output'], batch['desc']
      pose, trajectory, start_trajectory = X
      pose_gt, trajectory_gt, start_trajectory_gt = Y

      x = torch.cat((trajectory, pose), dim=-1)
      y = torch.cat((trajectory_gt, pose_gt), dim=-1)

      x = x.to(device)
      y = y.to(device)
      if isinstance(s2v, torch.Tensor):
        s2v = s2v.to(device)

      ## Transform before the model
      x = pre.transform(x)
      y = pre.transform(y)

      if desc == 'train':
        y_cap, internal_losses = model(x, s2v, train=True)
      else:
        y_cap, internal_losses = model(x, s2v, train=False)
        
      loss = 0
      loss_ = 0
      if y_cap is not None: ## if model returns only internal losses
        loss = criterion(y_cap, y)
        loss_ = loss.item()

      for i_loss in internal_losses:
        loss += i_loss
        loss_ += i_loss.item()
        running_internal_loss += i_loss.item()

      running_count +=  np.prod(y.shape)    
      running_loss += loss_
      ## update tqdm
      Tqdm.set_description(desc+' {:.4f} {:.4f}'.format(running_loss/running_count, running_internal_loss/running_count))
      Tqdm.refresh()
      
      if desc == 'train':
        loss.backward()
        optim.step()

      x = x.detach()
      y = y.detach()
      loss = loss.detach()
      if y_cap is not None:
        y_cap = y_cap.detach()
      internal_losses = [i.detach() for i in internal_losses]
      if count>=0 and args.debug: ## debugging by overfitting
        break

    return running_loss/running_count

  num_epochs = args.num_epochs

  ## set up curriculum learning for training
  time_list = []
  time_list_idx = 0
  if curriculum:
    for power in range(1, int(np.log2(time-1)) + 1):
      time_list.append(2**power)
    data.update_dataloaders(time_list[0])
  time_list.append(time)
  tqdm.write('Training up to time: {}'.format(time_list[time_list_idx]))
    
  ## Training Loop
  for epoch in tqdm(range(num_epochs), ncols=20):
    train_loss = loop(model, train, pre, 'train', epoch)
    dev_loss = loop(model, dev, pre, 'dev', epoch)
    test_loss = loop(model, test, pre, 'test', epoch)
    scheduler.step() ## Change the Learning Rate
    
    ## save results
    book.update_res({'train':train_loss,
                     'dev':dev_loss,
                     'test':test_loss})
    book._save_res()

    ## update tensorboard
    book.update_tb({'scalar':[[f'{args.cpk}/train', train_loss, epoch],
                              [f'{args.cpk}/dev', dev_loss, epoch],
                              [f'{args.cpk}/test', test_loss, epoch]]})
                   
                   # 'histogram':[[f'{args.cpk}/'+name, param.clone().cpu().detach().numpy(), epoch]
                    #             for name, param in model.named_parameters()]})

    ## print results
    book.print_res(epoch, key_order=['train','dev','test'], exp=exp_num, lr=scheduler.get_lr())

    if book.stop_training(model, epoch):
      ## if early_stopping criterion is met,
      ## start training with more time steps
      time_list_idx += 1
      book.stop_count = 0 ## reset the threshold counter
      book.best_dev_score = np.inf
      model.load_state_dict(copy.deepcopy(book.best_model))
      if len(time_list) > time_list_idx:
        time_ = time_list[time_list_idx]
        data.update_dataloaders(time_)
        tqdm.write('Training up to time: {}'.format(time_))
      else:
        break
    
  ## Sample
  print('Loading the best model and running the sample loop')
  args.__dict__.update({'load':book.name(book.weights_ext[0], book.weights_ext[1], args.save_dir)})
  sample(args, exp_num, data)

  ## Render (on a cpu only node)
  # feats_kind_dict = {'rifke':'fke'}
  # print('Rendering')
  # render = Slurm('render', slurm_kwargs={'partition':'cpu_long', 'time':'10-00:00', 'n':10})
  # python_cmd = ['source activate torch',
  #               'python render.py -dataset {} -load {} -feats_kind {} -render_list {}'.format(
  #                 args.dataset,
  #                 args.load,
  #                 feats_kind_dict[args.feats_kind],
  #                 args.render_list)]
  # render.run('\n'.join(python_cmd))

  ## Render new sentences
  # print('Rendering New Sentences')
  # render_new_sentences(args, exp_num, data)

  # End Log
  book._stop_log()

if __name__ == '__main__':
  argparseNloop(train)
