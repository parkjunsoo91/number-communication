import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from nac import NAC
from nalu import NALU
from nac import NeuralAccumulatorCell
from nalu import NeuralArithmeticLogicUnitCell
from torch.nn.parameter import Parameter


class BaselineReceiver(nn.Module):
    def __init__(self, hidden_size=32, vocab_size=10):
        super(BaselineReceiver, self).__init__()
        
        self.input_size = vocab_size+3

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.final = NALU(1, self.hidden_size, 1, 1)
        #self.final = nn.Linear(self.hidden_size, 1)
    def forward(self, onehots):
        output, states = self.lstm(onehots)
        #print(output.size()) #batch * seq * dim
        summed = torch.sum(output, 1)
        
        out = self.final(summed)
        return out




class NNtester(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=100, vocab_size=10):
        super(NNtester, self).__init__()
        self.vocab_size = vocab_size
        vocab_dim = vocab_size+3
        
        self.generator = nn.Sequential(
            nn.Linear(in_dim*5, hidden_dim),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(hidden_dim, vocab_dim),
        )
        #self.softmax = nn.Softmax(dim=1)
    def forward(self, x, onehots):
        device = x.device

        ones = torch.ones(x.size()).to(device)
        x1 = torch.exp(x)
        x2 = torch.fmod(x, self.vocab_size)
        x4 = torch.pow(x, 2)
        x5 = torch.pow(x, 3)
        comb = torch.cat((x,x1,x2,x4,x5),dim=1)
        #print(x)
        #comb = torch.cat((x,x,x,x,x),dim=1)
        s = self.generator(comb)
        
        return [s]

class NumToLang_LSTM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, max_len):
        super(NumToLang_LSTM, self).__init__()

        self.hidden_size = hidden_dim
        self.onehot_size = vocab_size
        self.embedding_size = 10
        self.gen_dim = 100
        self.max_len = max_len

        self.expander_h = nn.Sequential(
            nn.Linear(1, self.hidden_size)
        )
        self.expander_c = nn.Sequential(
            nn.Linear(1, self.hidden_size)
        )
        #self.embedding = nn.Embedding(self.onehot_size, self.embedding_size)
        self.lstm = nn.LSTM(self.onehot_size, self.hidden_size, batch_first=True)
        self.hidden2word = nn.Sequential(
            nn.Linear(self.hidden_size, self.gen_dim),
            nn.ReLU(),
            nn.Linear(self.gen_dim, self.onehot_size),
            #nn.Linear(self.hidden_size, self.onehot_size),
            nn.LogSoftmax(dim=2)
        )
        self.hiddens = [torch.zeros(1,1)]

    
    def forward(self, input, onehots, digits, teacher):
        device = input.device
        vocab_list = []

        h_0 = self.expander_h(input).unsqueeze(0)
        c_0 = self.expander_c(input).unsqueeze(0)

        if teacher:
        #if False:
            output, (hn, cn) = self.lstm(onehots, (h_0, c_0))
            tokens = self.hidden2word(output)
            tokens = torch.transpose(tokens, 0, 1)
            for i in range(tokens.size()[0]):
                vocab_list.append(tokens[i])
        else:
            
            h_n = h_0
            c_n = c_0
            input_n = onehots[:,0:1,:]
            for i in range(self.max_len):
                lstm_output, (h_n, c_n) = self.lstm(input_n, (h_n, c_n))
                token_prob = self.hidden2word(lstm_output)
                token_prob = torch.transpose(token_prob, 0, 1)
                token_prob = torch.squeeze(token_prob)
                input_n = torch.eye(token_prob.size()[1])[torch.argmax(token_prob, dim=1)].to(device)
                input_n = input_n.unsqueeze(1)

                vocab_list.append(token_prob)
        return vocab_list

class NumToLang_Modulus(nn.Module):
    def __init__(self, hidden_dim, vocab_size, max_len, modulus):
        super(NumToLang_Modulus, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = 1 # hidden_dim
        self.gen_dim = 100
        self.embedding_dim = vocab_size
        self.max_len = max_len
        self.modulus = 10

        self.combinator_matrix = Parameter(torch.Tensor([[0.1,-0.1]]))
        self.register_parameter('combinator_matrix', self.combinator_matrix)

        #self.expander = nn.Linear(1, self.hidden_dim, bias=False)

        #self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embed2val = nn.Linear(self.embedding_dim, self.hidden_dim, bias=False)

        #self.combinator = nn.Linear(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.combinator = nn.Linear(self.hidden_dim*2, self.hidden_dim, bias=False)
        with torch.no_grad():
            self.combinator.weight = self.combinator_matrix
        self.generator = nn.Sequential(
                nn.Linear(self.hidden_dim*2, self.gen_dim),
                nn.ReLU(),
                nn.Linear(self.gen_dim, self.vocab_size),
                #nn.Linear(self.hidden_dim*2, vocab_size),
                nn.LogSoftmax(dim=1)
        )
        

    def forward(self, x, onehots, digits, teacher):
        #print('x', x.size())
        #print(x)
        batch_size = x.size()[0]
        device = x.device
        
        #hidden = self.expander(x)
        hidden = x
        input_digit = digits[:,0]
        vocab_outputs = []
        self.hiddens = []
        for i in range(self.max_len):

            #embed = self.embedding(input_digit)
            self.hiddens.append(hidden)
            mod = torch.fmod(hidden, self.modulus)
            
            # pow2 = torch.pow(mod,2)
            # pow3 = torch.pow(mod,3)
            # exp = torch.exp(mod)
            # materials = torch.cat((x, mod, pow2, pow3, exp),dim=1)
            word_prob = self.generator(torch.cat((mod,hidden),dim=1))

            # print('mod', mod)
            # print('word prob', word_prob.argmax(dim=1))
            
            vocab_outputs.append(word_prob)

            if i < self.max_len-1:
                if teacher:
                    input_digit = digits[:,i+1]
                else:   
                    input_digit = torch.argmax(word_prob, dim=1)
            
            #self.hiddens.append(input_digit.unsqueeze(-1))

            embed = torch.eye(self.vocab_size)[input_digit].to(device)
            val = self.embed2val(embed)

            cat = torch.cat((hidden, val), dim=1)
            #hidden = self.combinator(cat)
            hidden = (hidden-val) *0.1


        return vocab_outputs

class NumToLang_Extra(nn.Module):
    def __init__(self, hidden_dim, vocab_size, max_len):
        super(NumToLang_Extra, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = 1
        self.gen_dim = 100
        self.embedding_dim = vocab_size
        self.max_len = max_len

        self.ref_size = 1000
        self.ref1 = torch.Tensor([0,1,2,3,4,5,6,7,8,9, 1, 10, 100])
        self.nalu1 = NALU(1, len(self.ref1), 1, self.ref_size)
        self.nalu2 = NAC(1, self.hidden_dim + self.ref_size, 1, 1)
        self.gen = nn.Sequential(
                    nn.Linear(1, self.vocab_size),
                    nn.LogSoftmax(dim=1))

    def forward(self, x, onehots, digits, teacher):
        batch_size = x.size()[0]
        device = x.device
        vocab_outputs = []
        self.hiddens = []

        ref = self.nalu1(self.ref1.to(device))
        
        ref_rep = ref.repeat(batch_size, 1)
        hidden = x
        for i in range(self.max_len):
            self.hiddens.append(hidden)
            cat = torch.cat((hidden, ref_rep), dim=1)
            hidden = self.nalu2(cat)
            score = self.gen(hidden)
            vocab_outputs.append(score)
        
        return vocab_outputs


#num2lang format: 
# input: (input value, onehot sequence)
# output: list(vocab score)
class NumToLang_NALU(nn.Module):
    def __init__(self, hidden_dim, vocab_size, max_len):
        super(NumToLang_NALU, self).__init__()

        vocab_dim = vocab_size + 3 #adding eos token and padding token
        self.vocab_size = vocab_size
        self.vocab_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        gen_dim = 100

        #self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.generator = nn.Sequential(
                nn.Linear(5, gen_dim),
                nn.Tanh(),
                nn.Dropout(),
                nn.Linear(gen_dim, vocab_dim),
                nn.LogSoftmax(dim=1)
        )
        #self.nalu = NALU(num_layers=2, in_dim=hidden_dim+self.vocab_dim, hidden_dim=1+self.vocab_dim, out_dim=1)
        self.nalu = nn.Linear(self.vocab_dim+1, 1)

    def forward(self, x, onehots, digits):
        self.words = []
        batch_size = x.size()[0]
        device = x.device
        
        input_word = torch.zeros(batch_size, self.vocab_dim).to(device)
        #sender_hidden = self.linear1(x)
        #sender_hidden = x
        vocab_outputs = []
        
        for i in range(self.max_len):

            #mod = smooth_mod(x+0.01, self.vocab_size, 30)

            mod = torch.fmod(x, self.vocab_size)
            pow2 = torch.pow(mod,2)
            pow3 = torch.pow(mod,3)
            exp = torch.exp(mod)
            materials = torch.cat((x, mod, pow2, pow3, exp),dim=1)
            vocab_prob = self.generator(materials)
            #print(i, materials)
            #vocab_score = self.linear3(self.tanh(self.linear2(torch.cat((mod, sender_hidden),dim=1))))
            #vocab_score = self.linear2(torch.cat((mod, sender_hidden),dim=1))
            vocab_outputs.append(vocab_prob)
            if self.training:
                input_word = onehots[:,i,:]
            else:
                input_word = torch.eye(vocab_prob.size()[1])[torch.argmax(vocab_prob, dim=1)].to(device)

            x = self.nalu(torch.cat((x, input_word), dim=1))
            
        return vocab_outputs


class NumToLang(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=4, vocab_size=10, max_len=5, rnn='lstm'):
        super(NumToLang, self).__init__()

        vocab_dim = vocab_size + 2 #adding eos token and padding token
        self.vocab_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.rnn = rnn

        self.linear1 = nn.Linear(5, hidden_dim)
        self.grucell = nn.GRUCell(vocab_dim, hidden_dim)
        self.lstmcell = nn.LSTMCell(vocab_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_dim)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(vocab_dim, vocab_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, onehots):
        self.words = []
        batch_size = x.size()[0]
        device = x.device
        #print("digit size,",onehots.size())
        input_word = torch.zeros(batch_size, self.vocab_dim).to(device)
        mul = torch.mul(x, 10)
        pow2 = torch.pow(x,2)
        pow3 = torch.pow(x,3)
        exp = torch.mul(x,2)
        materials = torch.cat((x, mul, pow2, pow3, exp),dim=1)

        sender_hidden = self.linear1(materials)
        if self.rnn == 'lstm':
            c = torch.zeros(batch_size, self.hidden_dim).to(device)
        vocab_outputs = []
        for i in range(self.max_len):
            
            sender_hidden = self.grucell(input_word, sender_hidden)
            if self.rnn == 'lstm':
                sender_hidden, c = self.lstmcell(input_word, (sender_hidden, c))
            #vocab_score = self.linear3(self.relu(self.linear2(sender_hidden)))
            vocab_score = self.linear2(sender_hidden)
            vocab_prob = self.softmax(vocab_score)
            vocab_outputs.append(vocab_prob)
            if self.training:
                input_word = onehots[:,i,:]
            else:
                input_word = vocab_prob.detach()

        return vocab_outputs

class NumToLang_ff(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=100, vocab_size=10, max_len=5):
        super(NumToLang_ff, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        vocab_dim = vocab_size + 2
        self.vocab_dim = vocab_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear11 = nn.Linear(hidden_dim, vocab_dim*max_len)
        self.softmax = nn.Softmax()

    def forward(self, x, onehots):
        
        vocab_outputs = []
        hidden = self.linear1(x)
        outputs = self.linear11(hidden)
        for i in range(self.max_len):
            vocab_prob = self.softmax(outputs[:,i*self.vocab_dim:(i+1)*self.vocab_dim])
            vocab_outputs.append(vocab_prob)
        return vocab_outputs




class Numeral_Machine(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=4, vocab_dim=10, out_dim=1, tau=5):
        super(Numeral_Machine, self).__init__()
        self.reverse = True
        self.tau = tau
        self.DEU = Digit_Emitter_Unit()
        
        self.linear1 = nn.Linear(1, vocab_dim*10)
        self.linear2 = nn.Linear(vocab_dim*10, vocab_dim)

        self.receiver_unit = nn.Linear(11, 1)


    def forward(self, x):
        batch_size = x.size()[0]
        device = x.device
        self.words = []
        vocab_outputs = []
        receiver_hidden = torch.zeros(batch_size, 1).to(device)

        for i in range(8):
            x, modulo = self.DEU(x)
            vocab_hidden = torch.relu(self.linear1(modulo))
            vocab_score = self.linear2(vocab_hidden)
            vocab_prob = F.log_softmax(vocab_score, dim=1)

            if self.training:
                vocab_output = F.gumbel_softmax(vocab_prob, hard=True, tau=self.tau)
                self.words.append(torch.argmax(vocab_output, dim=1)) #()
            else:
                vocab_output = torch.eye(vocab_prob.size()[1])[torch.argmax(vocab_prob, dim=1)].to(device)
                self.words.append(torch.argmax(vocab_prob, dim=1))

            vocab_outputs.append(vocab_output)
            
        for t in range(len(vocab_outputs)):
            reverse = self.reverse
            index = len(vocab_outputs)-t-1 if reverse else t

            receiver_hidden = self.receiver_unit(torch.cat((vocab_outputs[index], receiver_hidden),dim=1))
        
        return receiver_hidden

        

class Digit_Emitter_Unit(nn.Module):
    def __init__(self, order=20):
        super(Digit_Emitter_Unit, self).__init__()
        self.order = order

        self.sub = NeuralAccumulatorCell(2, 2)
        self.div = NeuralArithmeticLogicUnitCell(2, 1)


    def forward(self, x):
        mod = smooth_mod(x[:,0] + 0.5, 5, self.order).unsqueeze(1)
        #mod = x%5
        h = self.sub(torch.cat((x, mod), dim=1))
        out = self.div(h)
        return out, mod

class Mod_Agent(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=2, out_dim=1, order=10):
        super(Mod_Agent, self).__init__()
        self.order = order
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(1, 1)

        nn.init.eye_(self.linear1.weight)
    def forward(self, x):
        # h = self.linear1(x)
        # h2 = smooth_mod(h[:,0], h[:,1], self.order).unsqueeze(1)
        # return h2

        #return self.linear2(h2)
        
        return smooth_mod(x[:,0] + 0.1, x[:,1], self.order).unsqueeze(1)

def smooth_mod(x, L, order):
    #x%L
    term = 0
    L = L
    x = x
    for n in range(1, order+1):
        term += torch.sin(torch.div(2 * math.pi * n * x, L)) / n 
    return L/2 - L/math.pi * term



class Gumbel_Agent(nn.Module):
    def __init__(self, in_dim=1, vocab_dim=100, out_dim=1, tau=1):
        super(Gumbel_Agent, self).__init__()
        self.tau = tau

        self.linear1 = nn.Linear(in_dim, vocab_dim)
        self.linear2 = nn.Linear(vocab_dim, out_dim)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        device = x.device
        self.words = []

        vocab_score = self.linear1(x)
        if self.training:
            vocab_output = F.gumbel_softmax(vocab_score, hard=True, tau=self.tau)
            self.words.append(torch.argmax(vocab_output, dim=1)) #()
        else:
            vocab_output = torch.eye(vocab_score.size()[1])[torch.argmax(vocab_score, dim=1)].to(device)
            self.words.append(torch.argmax(vocab_score, dim=1))
        
        output = self.linear2(vocab_output)
        return output

class Multi_Gumbel_Agent(nn.Module):
    def __init__(self, in_dim=1, vocab_dim=4, multi=8, out_dim=1, tau=1):
        super(Multi_Gumbel_Agent, self).__init__()
        self.tau = tau
        self.multi = multi
        self.vocab_dim = vocab_dim

        self.linear1 = nn.Linear(in_dim, vocab_dim*multi)
        self.linear2 = nn.Linear(vocab_dim*multi, out_dim)

        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, x):
        device = x.device
        self.words = []

        vocab_score = self.linear1(x)
        vocab_outputs = []
        for i in range(self.multi):
            if self.training:
                vocab_output = F.gumbel_softmax(vocab_score[:,self.vocab_dim*i:self.vocab_dim*(i+1)], hard=True, tau=self.tau)
                self.words.append(torch.argmax(vocab_output, dim=1)) #()
            else:
                vocab_output = torch.eye(self.vocab_dim)[torch.argmax(vocab_score[:,self.vocab_dim*i:self.vocab_dim*(i+1)], dim=1)].to(device)
                self.words.append(torch.argmax(vocab_output, dim=1))
            vocab_outputs.append(vocab_output)
        #print(self.words)
        vocab_outputs = torch.cat(vocab_outputs, dim=1)
        
        output = self.linear2(vocab_outputs)
        return output

class Recurrent_one(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=5, out_dim=1, seq_len=2, rnn_layers=1, rnn='rnn', reverse=False):
        super(Recurrent_one, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.rnn_layers = rnn_layers
        self.rnn = rnn
        self.reverse = reverse

        if rnn == "nac":
            self.linear1 = NeuralAccumulatorCell(in_dim, hidden_dim)
            self.linear3 = NeuralAccumulatorCell(hidden_dim, hidden_dim)
            self.linear4 = NeuralAccumulatorCell(hidden_dim, out_dim)
        elif rnn == 'nalu':
            self.linear1 = NeuralArithmeticLogicUnitCell(in_dim, hidden_dim)
            self.linear3 = NeuralArithmeticLogicUnitCell(hidden_dim, hidden_dim)
            self.linear4 = NeuralArithmeticLogicUnitCell(hidden_dim, out_dim)
        else:
            self.linear1 = nn.Linear(in_dim, hidden_dim)
            self.linear3 = nn.Linear(hidden_dim, hidden_dim)
            self.linear4 = nn.Linear(hidden_dim, out_dim)

        if self.rnn == "rnn":
            self.rnncell1 = nn.RNNCell(vocab_dim, hidden_dim, nonlinearity='relu')
            self.rnncell2 = nn.RNNCell(vocab_dim, hidden_dim, nonlinearity='relu')        
        elif self.rnn == "gru":
            self.rnncell1 = nn.GRUCell(vocab_dim, hidden_dim)
            self.rnncell2 = nn.GRUCell(vocab_dim, hidden_dim)
        elif self.rnn == 'nac':
            self.rnncell1 = NeuralAccumulatorCell(vocab_dim + hidden_dim, hidden_dim)
            self.rnncell2 = NeuralAccumulatorCell(vocab_dim + hidden_dim, hidden_dim)
        elif self.rnn == "nalu":
            self.rnncell1 = NeuralArithmeticLogicUnitCell(vocab_dim + hidden_dim, hidden_dim)
            self.rnncell2 = NeuralArithmeticLogicUnitCell(vocab_dim + hidden_dim, hidden_dim)
        else:
            print("RNN cell argument error")
    
    def forward(self, x):
        self.words = []
        batch_size = x.size()[0]
        device = x.device
        
        #initialize hidden states and initial input for sender+receiver RNN layers
        sender_hidden = self.linear1(x)
        input_word = torch.zeros(batch_size, self.vocab_dim).to(device)
        receiver_hidden = torch.zeros(batch_size, self.hidden_dim).to(device)
        vocab_outputs = []

        for t in range(self.seq_len):
            if self.rnn in ['nalu', 'nac']:
                sender_hidden = self.rnncell1(torch.cat((input_word, sender_hidden), dim=1))
            else:
                sender_hidden = self.rnncell1(input_word, sender_hidden)
            
            #record vocab use
            vocab_outputs.append(vocab_output)
            
            input_word = vocab_output.detach()
            
            
            if self.rnn in ['nalu', 'nac']:
                receiver_hidden = self.rnncell2(torch.cat((vocab_output, receiver_hidden), dim=1))
            else:
                receiver_hidden = self.rnncell2(vocab_output, receiver_hidden)

        for t in range(len(vocab_outputs)):
            reverse = self.reverse
            index = len(vocab_outputs)-t-1 if reverse else t

            if self.rnn in ['nalu', 'nac']:
                receiver_hidden = self.rnncell2(torch.cat((vocab_outputs[index], receiver_hidden),dim=1))
            else:
                receiver_hidden = self.rnncell2(vocab_outputs[index], receiver_hidden)

        receiver_hidden = self.linear3(receiver_hidden)
        if not self.rnn in ['nalu', 'nac']:
            receiver_hidden = torch.relu(receiver_hidden)
        scalar_output = self.linear4(receiver_hidden)
        #print("scalar output", scalar_output)
        
        return scalar_output


class Sender_Receiver(nn.Module):

    def __init__(self, in_dim=1, hidden_dim=5, out_dim=1, vocab_dim=10, seq_len=2, tau=5, rnn='rnn', reverse=False):
        super(Sender_Receiver, self).__init__()
        
        #hyperparameters
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.vocab_dim = vocab_dim
        self.seq_len = seq_len
        self.tau = tau
        self.rnn = rnn
        self.reverse = reverse

        #layers
        if rnn == "nac":
            self.linear1 = NeuralAccumulatorCell(in_dim, hidden_dim)
            self.linear2 = NeuralAccumulatorCell(hidden_dim, vocab_dim)
            self.linear3 = NeuralAccumulatorCell(hidden_dim, hidden_dim)
            self.linear4 = NeuralAccumulatorCell(hidden_dim, out_dim)
        elif rnn == 'nalu':
            self.linear1 = NeuralArithmeticLogicUnitCell(in_dim, hidden_dim)
            self.linear2 = NeuralArithmeticLogicUnitCell(hidden_dim, vocab_dim)
            self.linear3 = NeuralArithmeticLogicUnitCell(hidden_dim, hidden_dim)
            self.linear4 = NeuralArithmeticLogicUnitCell(hidden_dim, out_dim)
        else:
            self.linear1 = nn.Linear(in_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, vocab_dim)
            self.linear3 = nn.Linear(hidden_dim, hidden_dim)
            self.linear4 = nn.Linear(hidden_dim, out_dim)

        #alternative design option: task-specific layers for multitask leraning
        # 1) (default) share the rnn receiver and have separate FF task specific layers
        # 2) have separate rnn receivers for each task

        if self.rnn == "rnn":
            self.rnncell1 = nn.RNNCell(vocab_dim, hidden_dim, nonlinearity='relu')
            self.rnncell2 = nn.RNNCell(vocab_dim, hidden_dim, nonlinearity='relu')        
        elif self.rnn == "gru":
            self.rnncell1 = nn.GRUCell(vocab_dim, hidden_dim)
            self.rnncell2 = nn.GRUCell(vocab_dim, hidden_dim)
        elif self.rnn == 'nac':
            self.rnncell1 = NeuralAccumulatorCell(vocab_dim + hidden_dim, hidden_dim)
            self.rnncell2 = NeuralAccumulatorCell(vocab_dim + hidden_dim, hidden_dim)
        elif self.rnn == "nalu":
            self.rnncell1 = NeuralArithmeticLogicUnitCell(vocab_dim + hidden_dim, hidden_dim)
            self.rnncell2 = NeuralArithmeticLogicUnitCell(vocab_dim + hidden_dim, hidden_dim)
        else:
            print("RNN cell argument error")

    def forward(self, x):
        #print("forward: input size is ", x.size()) # --> (batchsize)
        self.words = []
        batch_size = x.size()[0]
        device = x.device
        
        #initialize hidden states and initial input for sender+receiver RNN layers
        sender_hidden = self.linear1(x)
        input_word = torch.zeros(batch_size, self.vocab_dim).to(device)
        receiver_hidden = torch.zeros(batch_size, self.hidden_dim).to(device)
        vocab_outputs = []

        for t in range(self.seq_len):
            if self.rnn in ['nalu', 'nac']:
                sender_hidden = self.rnncell1(torch.cat((input_word, sender_hidden), dim=1))
            else:
                sender_hidden = self.rnncell1(input_word, sender_hidden)
            #print(t, sender_hidden)

            vocab_score = self.linear2(sender_hidden)
            vocab_prob = F.log_softmax(vocab_score, dim=1)

            if self.training:
                vocab_output = F.gumbel_softmax(vocab_prob, hard=True, tau=self.tau)
                self.words.append(torch.argmax(vocab_output, dim=1)) #()
            else:
                vocab_output = torch.eye(vocab_prob.size()[1])[torch.argmax(vocab_prob, dim=1)].to(device)
                self.words.append(torch.argmax(vocab_prob, dim=1))
            
            #record vocab use
            vocab_outputs.append(vocab_output)
            
            input_word = vocab_output.detach()
            
            
            if self.rnn in ['nalu', 'nac']:
                receiver_hidden = self.rnncell2(torch.cat((vocab_output, receiver_hidden), dim=1))
            else:
                receiver_hidden = self.rnncell2(vocab_output, receiver_hidden)

        for t in range(len(vocab_outputs)):
            reverse = self.reverse
            index = len(vocab_outputs)-t-1 if reverse else t

            if self.rnn in ['nalu', 'nac']:
                receiver_hidden = self.rnncell2(torch.cat((vocab_outputs[index], receiver_hidden),dim=1))
            else:
                receiver_hidden = self.rnncell2(vocab_outputs[index], receiver_hidden)

        receiver_hidden = self.linear3(receiver_hidden)
        if not self.rnn in ['nalu', 'nac']:
            receiver_hidden = torch.relu(receiver_hidden)
        scalar_output = self.linear4(receiver_hidden)
        #print("scalar output", scalar_output)
        
        return scalar_output



class Legacy_Sender_Receiver(nn.Module):
    def __init__(self, args):
        super(Legacy_Sender_Receiver, self).__init__()

        #hyperparameters
        self.sender_hidden_size = args.hidden
        self.receiver_hidden_size = args.hidden
        self.vocab_size = args.vocab + 1
        self.max_seq_len = args.seq
        self.hard = True
        self.tau = args.tau
        self.rnn = 'rnn' if args.rnn == None else args.rnn
        self.reverse = args.reverse

        if args.input == 'continuous' or args.input == None:
            self.input_size = 1
        elif args.input == 'discrete':
            self.input_size = args.max
        elif args.input == 'combined':
            self.input_size = args.max + 1
        self.input_type = args.input

        
        #layers
        self.linear1 = nn.Linear(self.input_size, self.sender_hidden_size)
        self.linear2 = nn.Linear(self.sender_hidden_size, self.vocab_size)

        #alternative design option: task-specific layers for multitask leraning
        # 1) (default) share the rnn receiver and have separate FF task specific layers
        # 2) have separate rnn receivers for each task
        
        if self.rnn == "gru":
            self.rnncell1 = nn.GRUCell(self.vocab_size, self.sender_hidden_size)
            self.rnncell2 = nn.GRUCell(self.vocab_size, self.receiver_hidden_size)
        elif self.rnn == "rnn":
            self.rnncell1 = nn.RNNCell(self.vocab_size, self.sender_hidden_size, nonlinearity='relu')
            self.rnncell2 = nn.RNNCell(self.vocab_size, self.receiver_hidden_size, nonlinearity='relu')
        elif self.rnn == "nalu":
            self.rnncell1 = NALU(1, self.vocab_size + self.sender_hidden_size, 0, self.sender_hidden_size)
            self.rnncell2 = NALU(1, self.vocab_size + self.receiver_hidden_size, 0, self.receiver_hidden_size)
        elif self.rnn == 'nac':
            self.rnncell1 = NAC(1, self.vocab_size + self.sender_hidden_size, 0, self.sender_hidden_size)
            self.rnncell2 = NAC(1, self.vocab_size + self.receiver_hidden_size, 0, self.receiver_hidden_size)
        else:
            print("error")
        #one hidden layer for processing output
        self.linear5 = nn.Linear(self.receiver_hidden_size, self.receiver_hidden_size)

        #layers for regression
        self.linear3 = nn.Linear(self.receiver_hidden_size, 1)

        #layers for classification
        self.linear4 = nn.Linear(self.receiver_hidden_size, args.max)

    def integerize(self):
        for p in self.parameters():
            p.data = p.data.to(torch.long).to(torch.float)

    def receiver_test(self, words):
        #words : (seq, batch, vocab size)
        batch_size = words.size()[1]
        device = words.device

        receiver_hidden = torch.zeros(batch_size, self.receiver_hidden_size).to(device)
        for t in range(self.max_seq_len):
            receiver_hidden = self.rnncell2(words[t,:,:], receiver_hidden)
        
        receiver_hidden = self.linear5(receiver_hidden)
        receiver_hidden = torch.relu(receiver_hidden)
        regression_output = self.linear3(receiver_hidden)
        classification_output = self.linear4(receiver_hidden)
        return regression_output, classification_output

    def forward(self, x, x_hot):
        #print("forward: input size is ", x.size()) # --> (batchsize)
        self.words = []
        self.saved_probs = []
        batch_size = x.size()[0]
        device = x.device
        #combine input
        if self.input_type == 'continuous' or self.input_type == None:
            combined_x = x
        elif self.input_type == 'discrete':
            combined_x = x_hot
        elif self.input_type == 'combined':
            combined_x = torch.cat((x, x_hot), 1)
        else:
            print("error")
        
        #initialize hidden states and initial input for sender+receiver RNN layers
        sender_hidden = self.linear1(combined_x)
        input_word = torch.zeros(batch_size, self.vocab_size).to(device)
        receiver_hidden = torch.zeros(batch_size, self.receiver_hidden_size).to(device)
        vocab_outputs = []

        #RESTORE THE BASE MODEL MAN

        for t in range(self.max_seq_len):
            if self.rnn in ['nalu', 'nac']:
                sender_hidden = self.rnncell1(torch.cat((input_word, sender_hidden), dim=1))
            else:
                sender_hidden = self.rnncell1(input_word, sender_hidden)
            #print("step", t, "sender_hidden")
            #print(sender_hidden)
            vocab_score = self.linear2(sender_hidden)
            #print("vocab_score")
            #print(vocab_score)
            vocab_prob = F.log_softmax(vocab_score, dim=1)
            #print("vocab_prob")
            #print(vocab_prob)
            if self.training:
                vocab_output = F.gumbel_softmax(vocab_prob, hard=True, tau=self.tau)
                self.words.append(torch.argmax(vocab_output, dim=1)) #()
                self.saved_probs.append(torch.sum(torch.mul(vocab_output, vocab_prob), dim=1)) #(batch)
            else:
                vocab_output = torch.eye(vocab_prob.size()[1])[torch.argmax(vocab_prob, dim=1)].to(device)
                self.words.append(torch.argmax(vocab_prob, dim=1))
            
            #record vocab use
            vocab_outputs.append(vocab_output)
            
            input_word = vocab_output.detach()
            
            
            if self.rnn in ['nalu', 'nac']:
                receiver_hidden = self.rnncell2(torch.cat((vocab_output, receiver_hidden), dim=1))
            else:
                receiver_hidden = self.rnncell2(vocab_output, receiver_hidden)

        for t in range(len(vocab_outputs)):
            reverse = self.reverse
            index = len(vocab_outputs)-t-1 if reverse else t

            if self.rnn in ['nalu', 'nac']:
                receiver_hidden = self.rnncell2(torch.cat((vocab_outputs[index], receiver_hidden),dim=1))
            else:
                receiver_hidden = self.rnncell2(vocab_outputs[index], receiver_hidden)
            
        #     #EOS token
        #     #if torch.argmax(vocab_output) == self.vocab_size-1: #EOS is the last token

        #     ############### this part makes different sequence length
        #     if torch.argmax(vocab_output) == 0:#EOS is the first token
        #         break
            
        #     if t == self.max_seq_len - 1 and torch.argmax(vocab_output) != self.vocab_size-1:
        #         vocab_outputs.append(torch.eye(vocab_prob.size()[1])[self.vocab_size-1].unsqueeze(0).to(device))
        #     ############### shit

        # #print(vocab_outputs)
        # #print(vocab_outputs)
        # torch.cat(vocab_outputs)
        # for t in range(len(vocab_outputs)):
        #     receiver_hidden = self.rnncell2(vocab_outputs[len(vocab_outputs)-t-1], receiver_hidden)
        
        receiver_hidden = self.linear5(receiver_hidden)
        receiver_hidden = torch.relu(receiver_hidden)
        regression_output = self.linear3(receiver_hidden)
        classification_output = self.linear4(receiver_hidden)
        return regression_output, classification_output

class Sender_Receiver_old(nn.Module):
    def __init__(self, args):
        super(Sender_Receiver_old, self).__init__()
        
        self.sender_hidden_size = args.hidden
        self.receiver_hidden_size = args.hidden
        self.vocab_size = args.vocab
        self.max_seq_len = args.seq
        self.hard = True
        self.tau = args.tau
        
        self.task_size = 0
        self.output_hidden_size = self.receiver_hidden_size
        #self.i_h = nn.Linear(self.input_size, self.sender_hidden_size)
        self.i_h = nn.Linear(1, self.sender_hidden_size) #case of continous input only

        #sender regression
        self.s_r = nn.Linear(self.sender_hidden_size, 1)

        self.sender_grucell = nn.GRUCell(self.vocab_size, self.sender_hidden_size) #gru
        #self.sender_lstmcell = nn.LSTMCell(self.vocab_size, self.sender_hidden_size) #lstm

        # self.i2h = nn.Linear(self.vocab_size + self.sender_hidden_size, self.sender_hidden_size)
        # self.i2o = nn.Linear(self.vocab_size + self.sender_hidden_size, self.output_size)

        self.h_w = nn.Linear(self.sender_hidden_size, self.vocab_size)

        self.receiver_gru = nn.GRU(self.vocab_size, self.receiver_hidden_size, 1)
        self.h_o = nn.Linear(self.receiver_hidden_size, self.output_size)
        self.c_ho = nn.Linear(self.task_size + self.receiver_hidden_size * 2, self.output_hidden_size)
        self.ho_o = nn.Linear(self.output_hidden_size, self.output_size)
        self.ho_r = nn.Linear(self.output_hidden_size, 1)

        self.set_train()

    def set_train(self):
        self.train()
        self.eval_mode = False
    def set_eval(self):
        self.eval()
        self.eval_mode = True

    def input_to_sentence(self, x_hot):
        batch_size = x_hot.size()[0]
        h = self.i_h(x_hot)
        sr = self.s_r(h)
        #c = self.i_h(x_hot)
        c = torch.zeros_like(h) #if lstm
        input_word = torch.zeros(batch_size, self.vocab_size, device=x_hot.device)
        output_words = torch.zeros(self.max_seq_len, batch_size, self.vocab_size, device=x_hot.device)
        output_scores = torch.zeros(self.max_seq_len, batch_size, self.vocab_size, device=x_hot.device)
        for t in range(self.max_seq_len):
            h = self.sender_grucell(input_word, h)
            #h, c = self.sender_lstmcell(input_word, (h, c)) #if lstm
            output_score = self.h_w(h)
            output_scores[t] = F.log_softmax(output_score, dim=1)
            if self.eval_mode:
                output_word = torch.eye(output_score.size()[1])[torch.argmax(output_score, dim=1)].to(device=x_hot.device)
            else:
                output_word = F.gumbel_softmax(output_score, hard = True, tau = self.tau)
            output_words[t] = output_word
            input_word = output_word.detach()  #what if we don't detach?
            #input_word = output_word
        return output_words, output_scores, sr

    def sentence_to_hidden(self, sentence):
        batch_size = sentence.size()[1]
        h_0 = torch.zeros(1, batch_size, self.receiver_hidden_size, device=sentence.device)
        #output(seq_len, batch, num_directions*hidden_size)
        #h_n(layers*directions, batch, hidden_size)
        output, h_n = self.receiver_gru(sentence, h_0)
        h_n = h_n.view(batch_size, self.receiver_hidden_size)
        return output, h_n

    def forward(self, x1_hot, x2_hot, t_hot, x1_float, x2_float):
        #combined input
        #x1_input = torch.cat((x1_hot, x1_float.unsqueeze(-1)), dim=1)
        #x2_input = torch.cat((x2_hot, x2_float.unsqueeze(-1)), dim=1)
        
        #continuous input only
        x1_input = x1_float.unsqueeze(-1)
        x2_input = x1_float.unsqueeze(-1)
        
        #discrete input only
        #x1_input = x1_hot
        #x2_input = x2_hot

        sentence1, score1, sender_reg1 = self.input_to_sentence(x1_input)
        sentence2, score2, sender_reg2 = self.input_to_sentence(x2_input)
        #sentence: size(seq, batch, vocab)
        self.sent1 = sentence1
        self.sent2 = sentence2
        self.score1 = score1
        self.score2 = score2
        self.sender_reg1 = sender_reg1
        self.sender_reg2 = sender_reg2
        self.words1 = torch.transpose(torch.argmax(sentence1, dim=2), 0, 1)
        self.words2 = torch.transpose(torch.argmax(sentence2, dim=2), 0, 1)

        _, h1 = self.sentence_to_hidden(sentence1)
        _, h2 = self.sentence_to_hidden(sentence2)
        
        guess1_score = self.h_o(h1)
        guess2_score = self.h_o(h2)
        combined = torch.cat((t_hot, h1, h2), dim=1)
        output_hidden = torch.relu(self.c_ho(combined))
        output_score = self.ho_o(output_hidden)
        output_regress = self.ho_r(output_hidden)

        return output_score, guess1_score, guess2_score, output_regress, sender_reg1, sender_reg2



class Simple_RNN(nn.Module):
    def __init__(self, args):
        super(Simple_RNN, self).__init__()

        #hyperparameters
        self.sender_hidden_size = args.hidden
        self.receiver_hidden_size = args.hidden
        self.vocab_size = args.vocab
        self.max_seq_len = args.seq
        self.hard = True
        self.tau = args.tau

        #layers
        self.linear1 = nn.Linear(1, self.sender_hidden_size)
        self.grucell1 = nn.GRUCell(self.vocab_size, self.sender_hidden_size)
        self.linear2 = nn.Linear(self.sender_hidden_size, self.vocab_size)
        self.grucell2 = nn.GRUCell(self.vocab_size, self.receiver_hidden_size)
        self.linear3 = nn.Linear(self.receiver_hidden_size, 1)

    def forward(self, x):
        self.words = []
        sender_hidden = F.relu(self.linear1(x))

        receiver_hidden = sender_hidden
        prediction = self.linear3(receiver_hidden)
        return prediction



class TwoLayer(nn.Module):
    def __init__(self, args):
        super(TwoLayer, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, args.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden, 1),
        )
    def forward(self, x):
        return self.model(x)

class Simple_Linear(nn.Module):
    def __init__(self, args):
        super(Simple_Linear, self).__init__()
        self.linear1 = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear1(x)

