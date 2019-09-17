import torch
import torch.nn as nn
import gensim

import pdb
torch.backends.cudnn.enabled = False

class LSTMEncoder(nn.Module):
  def __init__(self, hidden_size):
    super(LSTMEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.w2v = Word2Vec()
    self.input_size = self.w2v.tokenizer.hidden_size
    self.dec = nn.LSTM(input_size=self.input_size,
                       hidden_size=hidden_size,
                       num_layers=2,
                       batch_first=True).double()

  def sort(self, x, reverse=False):
    return zip(*sorted([(x[i], i) for i in range(len(x))], reverse=reverse))
    
  def sortNpermute(self, x, mask):
    mask_sorted, perm = self.sort(mask.sum(dim=-1).cpu().numpy(), reverse=True)
    return x[list(perm)], list(mask_sorted), list(perm)

  def inverse_sortNpermute(self, x, perm):
    _, iperm = self.sort(perm, reverse=False)
    if isinstance(x, list):
      return [x_[list(iperm)] for x_ in x]
    else:
      return x[list(iperm)]
  
  def forward(self, sentences):
    x_orig, mask_orig = self.w2v(sentences)
    x, mask, perm = self.sortNpermute(x_orig, mask_orig)
    x = torch.nn.utils.rnn.pack_padded_sequence(x, mask, batch_first=True)

    ''' forward pass through lstm '''
    x, (h,m) = self.dec(x)

    ''' get the output at time_step=t '''
    h = self.inverse_sortNpermute(h[-1], perm)

    return h, None

class LSTMAttentionEncoder(nn.Module):
  def __init__(self, hidden_size):
    super(LSTMAttentionEncoder, self).__init__()
    self.hidden_size = hidden_size
    self.w2v = Word2Vec()
    self.input_size = self.w2v.tokenizer.hidden_size
    self.dec = nn.LSTM(input_size=self.input_size,
                       hidden_size=hidden_size,
                       num_layers=2,
                       batch_first=True).double()

  def sort(self, x, reverse=False):
    return zip(*sorted([(x[i], i) for i in range(len(x))], reverse=reverse))
    
  def sortNpermute(self, x, mask):
    mask_sorted, perm = self.sort(mask.sum(dim=-1).cpu().numpy(), reverse=True)
    return x[list(perm)], list(mask_sorted), list(perm)

  def inverse_sortNpermute(self, x, perm):
    _, iperm = self.sort(perm, reverse=False)
    if isinstance(x, list):
      return [x_[list(iperm)] for x_ in x]
    else:
      return x[list(iperm)]
  
  def forward(self, sentences):
    x_orig, mask_orig = self.w2v(sentences)
    x, mask, perm = self.sortNpermute(x_orig, mask_orig)
    x = torch.nn.utils.rnn.pack_padded_sequence(x, mask, batch_first=True)

    ''' forward pass through lstm '''
    x, (h,m) = self.dec(x)

    ''' inverse sort the outputs to match the original order '''
    x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

    ''' get the output at time_step=t '''
    x, h = self.inverse_sortNpermute([x, h[-1]], perm)

    return h, x, mask_orig
  
class BaseTokenizer():
  def __init__(self, vocab):
    self.vocab = vocab
    self.hidden_size = 300
    self._UNK = '_UNK'
    self._SEP = '_SEP'
    self.random_vec = torch.rand(self.hidden_size)
    self.zero_vec = torch.zeros(self.hidden_size)
    
  def tokenize(self, sentence):
    words_ = sentence.split(' ')

    ''' Lowercase all words '''
    words_ = [w.lower() for w in words_]

    ''' Add _UNK for unknown words '''
    words = []
    for word in words_:
      if word in self.vocab:
        words.append(word)
      else:
        words.append('_UNK')
    return words
    
class Word2Vec(nn.Module):
  '''
  Take a bunch of sentences and convert it to a format that Bert can process
  * Tokenize
  * Add _UNK for words that do not exist
  * Create a mask which denotes the batches
  '''
  def __init__(self, path2file='./s2v/GoogleNews-vectors-negative300.bin.gz'):
    super(Word2Vec, self).__init__()
    self.dummy_param = nn.Parameter(torch.Tensor([1]))
    self.model = gensim.models.KeyedVectors.load_word2vec_format(path2file, binary=True)
    print('Loaded Word2Vec model')
    
    # Load pre-trained model tokenizer (vocabulary)
    self.tokenizer = BaseTokenizer(self.model.vocab)
    
    # Tokenized input
    text = "Who was Jim Henson ? Jim Henson was a puppeteer"
    tokenized_text = self.tokenizer.tokenize(text)
    print('Tokenization example')
    print('{}  --->  {}'.format(text, tokenized_text))
    
  def __call__(self, x):
    self.device = self.dummy_param.device
    x = [self.tokenizer.tokenize(x_) for x_ in x]
    max_len = max([len(x_) for x_ in x])

    mask = torch.Tensor([[1]*len(x_) + [0]*(max_len-len(x_)) for x_ in x]).long().to(self.device)
    x = [x_ + ['_SEP']*(max_len-len(x_)) for x_ in x]
    vectors = []
    for sentence in x:
      vector = []
      for word in sentence:
        if word == self.tokenizer._UNK:
          vector.append(self.tokenizer.random_vec)
        elif word == self.tokenizer._SEP:
          vector.append(self.tokenizer.zero_vec)
        else:
          vector.append(torch.from_numpy(self.model.word_vec(word)))
      vector = torch.stack(vector, dim=0).double().to(self.device)
      vectors.append(vector)                                                           
    vectors = torch.stack(vectors, dim=0).double()
    return vectors, mask
