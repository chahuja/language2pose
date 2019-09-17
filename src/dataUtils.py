import pandas as pd
import re
import pdb

from data.data import *
from dataProcessing.meanVariance import loadMeanVariance
from dataProcessing.angDivisor import loadAngDivisor

from tqdm import tqdm
import json
import pickle as pkl
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
import string
import wordsegment

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, ConcatDataset

from pytorch_pretrained_bert import BertTokenizer

from common.quaternion import *

_FLOAT_EPS = 1e-8

class Data():
  def __init__(self, path2data, dataset, lmksSubset, desc=None, split=(0.8,0.1), batch_size=100, time=100, chunks=1, offset=0, shuffle=True, mask=[1, 0, 1, 1, 1, 1, 1], feats_kind='quaternion', s2v=False, load_data=True, f_new=4):
    if feats_kind=='quaternion':
      assert len(mask) == 7, 'quaternions have 4 params for root joint'
    elif feats_kind=='euler':
      assert len(mask) == 6, 'euler angles have 3 params for root joint'
    elif feats_kind == 'rifke':
      assert len(mask) == 1, 'Rotation invariant Forward Kinematics have 3 parameters for root joint'
    
    self.raw_data = eval(dataset)(path2data)
    self.df = self.raw_data._get_df()
    self.s2v = s2v
    self.desc = desc
    self.dataset = dataset
    
    ## Choose a subset from the dataset based on descriptions
    ## TODO add Regex support
    self.dfs = []
    
    if desc:
      self.vocab, self.rev_vocab = self.create_vocab()
      for desc_ in desc:
        L, U = desc_.lower(), desc_.upper()
        LU = ''.join(['{}{}'.format(l, u) for l, u in zip(L, U)])
        str_format = '[{}{}]'*len(L)
        str_format = str_format.format(*LU)
        p = re.compile('^{}'.format(str_format))
        fn = lambda x: True if p.search(x) else False
        df_ = self.df[self.df['descriptions'].apply(fn)]
        df_.loc[:, 'class'] = [desc_]*df_.shape[0]
        self.dfs.append(df_)

      self.df = pd.concat(self.dfs).sample(frac=1).reset_index(drop=True)
        
    # if self.s2v:
    #   self.file2id = pkl.load(open('s2v/{}_files.pkl'.format(dataset), 'rb'))
    #   self.vectors = np.load('s2v/{}_vectors.npy'.format(dataset))
      
    self.lmksSubset = lmksSubset
    self.split = split
    self.time = time
    self.chunks = chunks
    self.offset = offset
    self.mask = mask
    self.feats_kind = feats_kind

    f = self.raw_data._get_f()
    if not f_new:
      f_new = f
    self.f_ratio = int(f/f_new)
    
    if load_data:
      self.datasets = self.tdt_split()
      self.dataLoader_kwargs = {'batch_size':batch_size,
                                'shuffle':shuffle}

      self.update_dataloaders(time)

  def update_dataloaders(self, time):
    ## update idx_list for all minidata
    for key in self.datasets:
      for d_ in self.datasets[key].datasets:
        d_.update_idx_list(time)
    
    self.train = DataLoader(ConcatDataset(self.datasets['train'].datasets), **self.dataLoader_kwargs)
    self.dev = DataLoader(ConcatDataset(self.datasets['dev'].datasets), **self.dataLoader_kwargs)
    self.test = DataLoader(ConcatDataset(self.datasets['test'].datasets), **self.dataLoader_kwargs)

  def tdt_split(self):
    length = self.df.shape[0]
    end_train = int(length*self.split[0])
    start_dev = end_train
    end_dev = int(start_dev + length*self.split[1])
    start_test = end_dev

    df_train = self.df[:end_train]
    df_dev = self.df[start_dev:end_dev]
    df_test = self.df[start_test:]

    minidataKwargs = {'lmksSubset':self.lmksSubset,
                      'time':self.time,
                      'offset':self.offset,
                      'mask':self.mask,
                      'feats_kind':self.feats_kind,
                      'chunks':self.chunks,
                      'f_ratio':self.f_ratio,
                      'dataset':self.dataset}

    if self.desc:
      dataset_train = ConcatDataset([MiniData(row[self.feats_kind],
                                              sentence_vector=self.file2vec(desc=row['class']),
                                              **minidataKwargs) for i, row in tqdm(df_train.iterrows())])
      dataset_dev = ConcatDataset([MiniData(row[self.feats_kind],
                                            sentence_vector=self.file2vec(desc=row['class']),
                                            **minidataKwargs) for i, row in tqdm(df_dev.iterrows())])
      dataset_test = ConcatDataset([MiniData(row[self.feats_kind],
                                             sentence_vector=self.file2vec(desc=row['class']),
                                             **minidataKwargs) for i, row in tqdm(df_test.iterrows())])
    elif self.s2v:
      dataset_train = ConcatDataset([MiniData(row[self.feats_kind],
                                              sentence_vector=row['descriptions'],
                                              **minidataKwargs) for i, row in tqdm(df_train.iterrows()) if row['descriptions']])
      dataset_dev = ConcatDataset([MiniData(row[self.feats_kind],
                                            sentence_vector=row['descriptions'],
                                            **minidataKwargs) for i, row in tqdm(df_dev.iterrows()) if row['descriptions']])
      dataset_test = ConcatDataset([MiniData(row[self.feats_kind],
                                             sentence_vector=row['descriptions'],
                                             **minidataKwargs) for i, row in tqdm(df_test.iterrows()) if row['descriptions']])
    else:
      dataset_train = ConcatDataset([MiniData(row[self.feats_kind],
                                              sentence_vector=row['descriptions'],
                                              **minidataKwargs) for i, row in tqdm(df_train.iterrows())])
      dataset_dev = ConcatDataset([MiniData(row[self.feats_kind],
                                            sentence_vector=row['descriptions'],
                                            **minidataKwargs) for i, row in tqdm(df_dev.iterrows())])
      dataset_test = ConcatDataset([MiniData(row[self.feats_kind],
                                             sentence_vector=row['descriptions'],
                                             **minidataKwargs) for i, row in tqdm(df_test.iterrows())])
      
    return {'train':dataset_train,
            'dev':dataset_dev,
            'test':dataset_test}

  @staticmethod
  def tokenize(desc):
    wordsegment.load()
    desc = desc.lower()

    ## remove punctuation
    desc = nltk.tokenize.WordPunctTokenizer().tokenize(desc)
    exclude = set(string.punctuation)
    desc = [''.join([c for c in ch if c not in exclude]) for ch in desc]
    desc = [ch for ch in desc if ch]

    ## word segmentor
    desc = wordsegment.segment(' '.join(desc))
    
    ## remove stop words
    stopwords = set(nltk.corpus.stopwords.words('english'))
    desc = [ch for ch in desc if ch not in stopwords]

    ## remove integer values
    words = []
    for ch in desc:
      try:
        int(ch)
      except:
        words.append(ch)

        
    ## Lemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(word) for word in words] 
        
    return ' '.join(words)

  def file2vec(self, file_name=None, desc=None):
    if self.s2v:
      try:
        vec = self.vectors[self.file2id[file_name]]
      except:
        vec = np.random.rand(300,)
    elif self.desc:
      vec = np.zeros((len(self.vocab)))
      vec[self.vocab[desc]] = 1
    else:
      vec = None  
    return vec

  def create_vocab(self):
    vocab = {d:idx for idx, d in enumerate(self.desc)}
    rev_vocab = self.desc
    return vocab, rev_vocab
  
    # vocab = {}
    # reverse_vocab = []
    # for i, row in self.df.iterrows():
    #   desc = row['descriptions']
    #   words = Data.tokenize(desc)
    #   for word in words:
    #     if word not in reverse_vocab:
    #       vocab[len(reverse_vocab)] = word
    #       reverse_vocab.append(word)
    # return vocab, reverse_vocab
    
  @property
  def input_shape(self):
    return self.train.dataset.datasets[0].input_shape

  @property
  def columns(self):
    '''
    Returns all the columns and the subset of the columns used as the input
    '''
    return self.train.dataset.datasets[0].columns


class MiniData(Dataset):
  def __init__(self, path2csv, lmksSubset, time, offset=0, mask=[1, 0, 1, 1, 1, 1, 1], feats_kind='quaternion', sentence_vector=None, chunks=1, dataset='CMUMocap', f_ratio=30):
    super(MiniData, self).__init__()
    self.path2csv = path2csv
    self.mat_full = pd.read_csv(path2csv, index_col=0)

    ## if offset == 0; autoencoder
    ## if offset >= 1; prediction
    self.offset = offset
    self.mask = mask
    
    ## get subset_columns
    self.columns_ = list(self.mat_full.columns)

    self.sentence_vector = sentence_vector
    self.chunks = chunks

    self.f_ratio = f_ratio

    if feats_kind in {'euler', 'quaternion'}:
      self.columns_subset = Regex.get_columns_subset(self.columns_, lmksSubset, dataset=dataset)
    elif feats_kind in {'rifke'}:
      self.columns_subset = self.columns_[3:]

    if feats_kind == 'euler':
      self.root_columns = ['root_tx', 'root_ty', 'root_tz', 'root_rx', 'root_ry', 'root_rz']
    elif feats_kind == 'quaternion':
      self.root_columns = ['root_tx', 'root_ty', 'root_tz', 'root_rw', 'root_rx', 'root_ry', 'root_rz']
    elif feats_kind == 'rifke':
      self.root_columns = ['root_tx', 'root_ty', 'root_tz']
      self.root_columns = ['root_ty']
    
    self.mat = self.mat_full[self.columns_subset].values.astype(np.float64)
    self.root = self.mat_full[self.root_columns].values.astype(np.float64)

    ## update idx_list:: call this for MiniData to update data loader
    self.update_idx_list(time)
        
  def __len__(self):
    return len(self.idx_list)

  def __getitem__(self, idx):
    assert self.offset >=0, 'offset cannot be negative'
    ## TODO
    mask = np.array(self.mask) ## [tx, ty, tz, rx, ry, rz]
    mask = mask[np.newaxis, :]
    root_delta = self.root[self.idx_list[idx]][1:] - mask * self.root[self.idx_list[idx]][:-1]
    #zeros = self.root[self.idx_list[idx]][0][np.newaxis, :] * (1-mask)
    #root_delta = np.concatenate((zeros, root_delta), axis=0)

    mat_ = self.mat[self.idx_list[idx]][1:]

    if self.offset >= 1:
      item = {'input':[mat_[:, :-self.offset],
                       root_delta[:, :-self.offset],
                       self.root[self.idx_list[idx]][0]],
              'output':[mat_[:, self.offset:],
                        root_delta[:, self.offset:],
                        self.root[self.idx_list[idx]][self.offset]],
              'path':self.path2csv}
    else:
      item = {'input':[mat_,
                       root_delta,
                       self.root[self.idx_list[idx]][0]],
              'output':[mat_,
                        root_delta,
                        self.root[self.idx_list[idx]][0]],
              'path':self.path2csv}

    if self.sentence_vector is not None:
      item.update({'desc':self.sentence_vector})

    return item

  def update_idx_list(self, time):
    if time:
      big_chunks = (self.mat.shape[0]-1)//(self.f_ratio*time*self.chunks)
      self.idx_list = [range(i*(time*self.f_ratio)*self.chunks,
                             (i+1)*(time*self.f_ratio)*self.chunks+1,
                             self.f_ratio)
                       for i in range(big_chunks)]
    else: ## If time is None (Mainly for sampling)
      self.idx_list = [range(0, self.mat.shape[0], self.f_ratio)]
      
    if self.idx_list:
      if len(self.idx_list[-1]) <= self.offset:
        self.idx_list = self.idx_list[:-1]

  
  @property
  def input_shape(self):
    return {'pose_size':self.mat.shape[-1],
            'trajectory_size':self.root.shape[-1]}

  @property
  def columns(self):
    return self.columns_, self.columns_subset


## works with the previous versions of the code
class MiniData_old(Dataset):
  def __init__(self, path2csv, lmksSubset, time, offset=0):
    super(MiniData, self).__init__()
    self.path2csv = path2csv
    self.mat_full = pd.read_csv(path2csv, index_col=0)

    ## if offset == 0; autoencoder
    ## if offset >= 1; prediction
    self.offset = offset
    
    ## get subset_columns
    self.columns_ = list(self.mat_full.columns)
    self.columns_subset = Regex.get_columns_subset(self.columns_, lmksSubset)
    
    self.mat = self.mat_full[self.columns_subset].values

    if time:
      chunks = self.mat.shape[0]//time
      self.idx_list = [range(i*time, (i+1)*time) for i in range(chunks)]
    else: ## If time is None (Mainly for sampling)
      self.idx_list = [range(0, self.mat.shape[0])]

    if self.idx_list:
      if len(self.idx_list[-1]) <= self.offset:
        self.idx_list = self.idx_list[:-1]
        
  def __len__(self):
    return len(self.idx_list)

  def __getitem__(self, idx):
    assert self.offset >=0, 'offset cannot be negative'
    if self.offset >= 1:
      return self.mat[self.idx_list[idx]].transpose(1,0)[:, :-self.offset], self.mat[self.idx_list[idx]].transpose(1,0)[:, self.offset:]
    else:
      return self.mat[self.idx_list[idx]].transpose(1,0), self.mat[self.idx_list[idx]].transpose(1,0)

  @property
  def input_shape(self):
    return self.mat.shape[-1]

  @property
  def columns(self):
    return self.columns_, self.columns_subset

class Transforms():
  def __init__(self, transforms, columns_subset, seed, mask, feats_kind, dataset, f_new):
    self.transforms = transforms
    self.mask = mask
    for tr in transforms:
      if tr == 'zNorm':
        self.mean, self.variance = loadMeanVariance(columns_subset, seed, mask, feats_kind, dataset, f_new)
        self.mean = self.mean.reshape(1, 1, -1)
        self.variance = self.variance.reshape(1, 1, -1)
      elif tr == 'angNorm':
        self.angDivisor = loadAngDivisor(columns_subset, seed)
        self.angDivisor = self.angDivisor.reshape(1, 1, -1)
      elif tr == 'quat':
        pass
      elif tr == 'zNormTranslation':
        self.mean, self.variance = loadMeanVariance(columns_subset, seed, mask, feats_kind, dataset, f_new)
        self.mean = self.mean.reshape(1, 1, -1)
        self.variance = self.variance.reshape(1, 1, -1)        
      else:
        assert 0, 'Transform not found'

  def transform(self, x):
    reduceTo360 = lambda x: (x - torch.floor(x/360.0)*360)
    for tr in self.transforms:
      if tr == 'zNorm':
        x = (x - self.mean.to(x.device))/(_FLOAT_EPS+self.variance.to(x.device)**0.5)
      elif tr == 'angNorm':
        x[:, :, 3:6] = reduceTo360(x[:, :, 3:6]) ## hardcoded the translation is the first 3 dimensions
        #x = x/self.angDivisor.to(x.device)
      elif tr == 'quat':
        x_temp = euler2quat(x[:, :, 3:6], order='xyz', deg=True) ## hadrcoded to only do root orientation
        x = torch.cat((x[:, :, :3], x_temp, x[:, :, 6:]), dim=-1)
      elif tr == 'zNormTranslation':
        x_ = x.clone()
        x_[..., :3] = (x[..., :3] - self.mean[..., :3].to(x.device))/(_FLOAT_EPS+self.variance[..., :3].to(x.device)**0.5)
        x_[..., 3:] = (x[..., 3:] - self.mean[..., 3:].to(x.device))
        x = x_
    return x

  def inv_transform(self, x):
    inv_transforms = list(reversed(self.transforms))
    for tr in inv_transforms:
      if tr == 'zNorm':
        x = (x*(self.variance.to(x.device)**0.5)) + self.mean.to(x.device)
      elif tr == 'angNorm':
        #x = x*self.angDivisor.to(x.device)
        pass
      elif tr == 'quat':
        x_temp = x[:, :, 3:7].clone()
        x_temp = x_temp/torch.norm(x_temp, dim=-1, keepdim=True)
        x_temp = qeuler(x_temp, order='xyz', deg=True)
        x = torch.cat((x[:, :, :3], x_temp, x[:, :, 7:]), dim=-1)
      elif tr == 'zNormTranslation':
        x_ = x.clone()
        x_[..., :3] = (x[..., :3]*(self.variance[..., :3].to(x.device)**0.5)) + self.mean[..., :3].to(x.device)
        x_[..., 3:] = (x[..., 3:] + self.mean[..., 3:].to(x.device))
        x = x_
    return x
  
class Regex():
  def __init__(self, columns, columns_re_dict={}, columns_re_subset=[]):
    self.columns = columns
    self.columns_re_dict = columns_re_dict
    self.columns_re_subset = columns_re_subset

  def __call__(self):
    output_dict = {}
    if self.columns_re_subset:
      for columns_re in self.columns_re_subset:
        columns_subset = set()
        if columns_re in self.columns_re_dict:
          for p in self.columns_re_dict[columns_re]:
            for col in self.columns:
              if p.search(col):
                columns_subset.add(col)
          output_dict.update({columns_re: sorted(list(columns_subset))})
    else:
      output_dict = {'all':self.columns}
    return output_dict

  @staticmethod
  def get_columns_subset(columns, lmksSubset, dataset='CMUMocap'):
    if dataset == 'CMUMocap':
      skelpath = 'skeleton/joints.py'
    elif dataset == 'KITMocap':
      skelpath = 'skeleton/kit_joints.py'
    else:
      assert 0, 'Dataset "{}" not found'.format(dataset)
      
    columns_dict = Regex(columns, load_config(skelpath), lmksSubset)()
    columns_subset = []
    for subset in columns_dict:
      columns_subset += columns_dict[subset]
    columns_subset = sorted(list(set(columns_subset)))
    return columns_subset
  
def load_config(conv_config):
  import importlib.util as U
  spec = U.spec_from_file_location('config_loader', conv_config)
  config_loader = U.module_from_spec(spec)
  spec.loader.exec_module(config_loader)
  return config_loader.conv_config

def get_columns(feats_kind, data):
  if feats_kind == 'euler':
    columns = ['root_tx', 'root_ty', 'root_tz', 'root_rx', 'root_ry', 'root_rz'] + data.columns[1]
  elif feats_kind == 'quaternion': ## transforms for columns with quat
    columns = ['root_tx', 'root_ty', 'root_tz', 'root_rw', 'root_rx', 'root_ry', 'root_rz'] + data.columns[1]
  elif feats_kind == 'rifke':
    columns = ['root_tx', 'root_ty', 'root_tz'] + data.columns[1]
    columns = ['root_ty'] + data.columns[1]
  return columns
