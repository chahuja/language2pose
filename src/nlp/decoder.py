import torch
import torch.nn as nn
import pdb
torch.backends.cudnn.enabled = False

class LSTMDecoder(nn.Module):
  def __init__(self, hidden_size, vocab_size):
    super(LSTMDecoder, self).__init__()
    self.hidden_size = hidden_size
    self.dec = nn.LSTM(input_size=hidden_size,
                       hidden_size=hidden_size,
                       num_layers=1,
                       batch_first=True)
    self.lin = nn.Linear(in_features=hidden_size,
                         out_features=vocab_size)

  def forward(self, h, time_steps):
    h = h.unsqueeze(0)
    x = torch.rand(h.shape[1], 1, self.hidden_size).to(h.device).double()
    m = torch.zeros_like(h).to(h.device).double()

    X = []
    for t in range(time_steps):
      x, (h, m) = self.dec(x, (h, m))
      X.append(x)
    x = torch.cat(X, dim=1)
    x = self.lin(x)

    return x

  def sample(self, h, time_steps):
    return self.forward(h, time_steps)
