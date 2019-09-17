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
from sample_seq2seq import *
from train_wordConditioned import train as train_wordConditioned

import numpy as np
from tqdm import tqdm

def train(args, exp_num):
  args_subset = ['exp', 'cpk', 'model', 'time']
  book = BookKeeper(args, args_subset, args_dict_update={},
                    tensorboard=args.tb)
  args = book.args
  global ARGS
  ARGS = args

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
  chunks = args.chunks
  offset = args.offset
  mask = args.mask
  feats_kind = args.feats_kind
  s2v = args.s2v
  f_new = args.f_new
  curriculum = args.curriculum
  kl_anneal = args.kl_anneal
  
  ## Load data iterables
  data = Data(path2data, dataset, lmksSubset, desc,
              split, batch_size=batch_size,
              time=time,
              chunks=chunks,
              offset=offset,
              shuffle=True,
              mask=mask,
              feats_kind=feats_kind,
              s2v=s2v,
              f_new=f_new)

  print('Data Loaded')
  
  ## Create a model
  device = torch.device('cuda:{}'.format(args.cuda)) if args.cuda>=0 else torch.device('cpu')
  input_shape = data.input_shape
  kwargs_keys = ['pose_size', 'trajectory_size']
  modelKwargs = {key:input_shape[key] for key in kwargs_keys}
  modelKwargs.update(args.modelKwargs)

  model = eval(args.model)(**modelKwargs)
  model.to(device).double()

  book._copy_best_model(model)
  print('Model Created')
    
  ## Load model
  if args.load:
    print('Loading Model')
    book._load_model(model)

  ## Loss function
  criterion = Loss(args.losses, args.lossKwargs)
  
  ## Optimizers
  optim = torch.optim.Adam(model.parameters(), lr=args.lr)
  #optim = torch.optim.RMSprop(model.parameters(), lr=args.lr)
  ## LR scheduler
  scheduler = lr_scheduler.ExponentialLR(optim, gamma=0.99)
  
  ## Transforms
  columns = get_columns(feats_kind, data)
  pre = Transforms(args.transforms, columns, args.seed, mask, feats_kind, dataset, f_new)

  def loop(model, data, pre, desc='train', epoch=0):
    running_loss = 0
    running_internal_loss = 0
    running_count = 0
    # if kl_anneal > 0:
    #   kl_weight = lambda x: min((x+1)/(kl_anneal+1.), 2)
    # else:
    #   kl_weight = lambda x: 1
    count = 0
    if desc == 'train':
      model.train(True)
    else:
      model.eval()
    
    Tqdm = tqdm(data, desc=desc+' {:.4f}'.format(running_loss/(count+1.)), leave=False, ncols=20)
    for count, batch in enumerate(Tqdm):
      model.zero_grad()
      optim.zero_grad()
      
      X, Y = batch['input'], batch['output']
      pose, trajectory, start_trajectory = X
      pose_gt, trajectory_gt, start_trajectory_gt = Y

      x = torch.cat((trajectory, pose), dim=-1)
      y = torch.cat((trajectory_gt, pose_gt), dim=-1)
      
      x = x.to(device)
      y = y.to(device)

      ## Transform before the model
      x = pre.transform(x)
      y = pre.transform(y)

      if desc=='train':
        y_cap, internal_losses = model(x, train=True)
      else:
        y_cap, internal_losses = model(x, train=False)

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
        
      # loss = criterion(y_cap, y)
      # loss_= loss.item()
      # #if count == 0 and desc == 'train':
      # #  pdb.set_trace()
      # for i_loss in internal_losses:
      #   loss += kl_weight(epoch) * i_loss
      #   loss_ += i_loss.item()
      #   running_internal_loss += i_loss
      
      # #running_loss += loss.item()
      # running_loss += loss_

      ## update tqdm
      Tqdm.set_description(desc+' {:.4f} {:.4f}'.format(running_loss/running_count, running_internal_loss/running_count))
      Tqdm.refresh()
      
      if desc == 'train':
        loss.backward()
        optim.step()
        # if kl_anneal == 0:
        #   y_cap, internal_losses = model(x, train=True, epoch=epoch)
        #   sum(internal_losses).backward()
        #   optim.step()

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
  
  ## Training Loop
  for epoch in tqdm(range(num_epochs), ncols=20):
    train_loss = loop(model, data.train, pre, 'train', epoch)
    dev_loss = loop(model, data.dev, pre, 'dev')
    test_loss = loop(model, data.test, pre, 'test')
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

    ## ignore increasing dev loss till the annealing occurs
    # if epoch < kl_anneal:
    #   book.stop_count = 0
    
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
    
  # End Log
  book._stop_log()

  # ## Sample
  print('Loading the best model and training with language input as well')
  args.__dict__.update({'load':book.name(book.weights_ext[0], book.weights_ext[1], args.save_dir),
                        'model':'Seq2SeqConditioned10'})
  train_wordConditioned(args, exp_num, data)

if __name__ == '__main__':
  argparseNloop(train)
