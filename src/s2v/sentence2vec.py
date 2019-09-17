'''
Saves fast text vectors for the dataset description in 
s2v/{dataset}_files.pkl and s2v/{dataset}_vectors.npy where the pickle file contains a dictionary pointing to the ids of the vectors
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from dataUtils import *
import numpy as np
from gensim.models.wrappers import FastText 
from argsUtils import argparseNloop
import pdb
from tqdm import tqdm
import json
import pickle as pkl

class S2V():
  def __init__(self):
    self.s2v_model = FastText.load_fasttext_format('fastTextModels/wiki.en.bin')
    print('Loaded FastText model')

  def description2vec(self, desc):
    return self.sentece2vec(Data.tokenize(desc))

  def sentece2vec(self, sentence):
    try:
      vec = self.s2v_model[sentence]
    except: ## TODO hardcoded to 300
      vec = np.random.rand(300)
    return vec

def loop(args, exp_num):
  path2data = args.path2data
  dataset = args.dataset
  lmksSubset = args.lmksSubset
  split = (args.train_frac, args.dev_frac)

  batch_size = args.batch_size
  time = args.time
  global chunks
  chunks = args.chunks
  offset = args.offset
  mask = args.mask
  feats_kind = args.feats_kind
  s2v = args.s2v

  data = Data(path2data, dataset, lmksSubset, None,
              split, batch_size=batch_size,
              time=time,
              chunks=chunks,
              offset=offset,
              shuffle=True,
              mask=mask,
              feats_kind=feats_kind,
              s2v=s2v,
              load_data=False)

  s2v = S2V()
  
  file_id = {}
  vectors = np.zeros((data.df.shape[0], 300))
  for idx, row in tqdm(data.df.iterrows()):
    file_id[row['euler']] = idx
    vectors[idx] = s2v.description2vec(row['descriptions'])

  pkl.dump(file_id, open('s2v/{}_files.pkl'.format(dataset), 'wb'))
  np.save('s2v/{}_vectors.npy'.format(dataset), vectors)
  print('Saved description vectors')
  
if __name__ == '__main__':
  argparseNloop(loop)



