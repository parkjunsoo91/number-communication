
import torch
from torch import LongTensor
from torch.nn import Embedding, LSTM
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


vocab = ['a','b','c','d','e','f']
embed = Embedding(len(vocab), 4) # embedding_dim = 4


seq_tensor = torch.tensor([[1,2],
                           [3,4],
                           [0,5]])
e = embed(seq_tensor)
print(e)