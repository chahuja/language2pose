import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

import pdb

def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off
  
class BertForSequenceEmbedding(nn.Module):
  def __init__(self, hidden_size):
    #config = BertConfig(32000) ## Dummy config file
    #super(BertForSequenceEmbedding, self).__init__(config)
    super(BertForSequenceEmbedding, self).__init__()
    self.hidden_size = hidden_size
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size, hidden_size),
                                    nn.Dropout(self.bert.config.hidden_dropout_prob),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size))
    ''' Fix Bert Embeddings and encoder '''
    toggle_grad(self.bert.embeddings, False)
    toggle_grad(self.bert.encoder, False)
    self.bert.eval()
    
    self.pre = BertSentenceBatching()
  
  def forward(self, sentences):
    input_ids, attention_mask = self.pre(sentences)
    token_type_ids = None
    outputs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    outputs = self.classifier(outputs[:, 0])
    return outputs, pooled_output

  def train_params(self, idx=0):
    params = [self.classifier, self.bert.pooler]
    return params[idx].parameters()

  def train(self, mode=True):
    self.training = mode
    for module in self.children():
      module.train(mode)
    self.bert.eval() ## bert needs to be in eval mode for both modes
    return self

class BertForAttentionSequenceEmbedding(nn.Module):
  def __init__(self, hidden_size):
    super(BertForAttentionSequenceEmbedding, self).__init__()
    self.hidden_size = hidden_size
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size, hidden_size),
                                    nn.Dropout(self.bert.config.hidden_dropout_prob),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size))
    ''' Fix Bert Embeddings and encoder '''
    toggle_grad(self.bert.embeddings, False)
    toggle_grad(self.bert.encoder, False)
    self.bert.eval()
    
    self.pre = BertSentenceBatching()
  
  def forward(self, sentences):
    input_ids, attention_mask = self.pre(sentences)
    token_type_ids = None
    outputs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    outputs = self.classifier(outputs)
    return outputs[:, 0, :], outputs, attention_mask

  def train_params(self, idx=0):
    params = [self.classifier, self.bert.pooler]
    return params[idx].parameters()

  def train(self, mode=True):
    self.training = mode
    for module in self.children():
      module.train(mode)
    self.bert.eval() ## bert needs to be in eval mode for both modes
    return self  

class BertSentenceBatching(nn.Module):
  '''
  Take a bunch of sentences and convert it to a format that Bert can process
  * Tokenize
  * Add [CLS] and [SEP] tokens
  * Create a mask which denotes the batches
  '''
  def __init__(self):
    super(BertSentenceBatching, self).__init__()
    self.dummy_param = nn.Parameter(torch.Tensor([1]))
    # Load pre-trained model tokenizer (vocabulary)
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenized input
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = self.tokenizer.tokenize(text)
    print('Tokenization example')
    print('{}  --->  {}'.format(text, tokenized_text))
    
  def __call__(self, x):
    self.device = self.dummy_param.device
    x = [self.tokenizer.tokenize(x_) for x_ in x]
    x = [['[CLS]'] + x_ + ['[SEP]'] for x_ in x]
    max_len = max([len(x_) for x_ in x])

    mask = torch.Tensor([[1]*len(x_) + [0]*(max_len-len(x_)) for x_ in x]).long().to(self.device)
    x = [x_ + ['[SEP]']*(max_len-len(x_)) for x_ in x]
    indexed_tokens = torch.Tensor([self.tokenizer.convert_tokens_to_ids(x_) for x_ in x]).long().to(self.device)
    return indexed_tokens, mask
