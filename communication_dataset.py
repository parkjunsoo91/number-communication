import numpy as np
import random
import torch

torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)


class Num2NumDataset(torch.utils.data.Dataset):
    def __init__(self, data_array, base, max_len, reverse):
        BASE = base
        MAX_LEN = max_len
        VOCAB_SIZE = 13
        self.vocabulary = ['0','1','2','3','4','5','6','7','8','9','SOS','EOS','PAD']
        self.SOS = self.vocabulary.index('SOS')
        self.EOS = self.vocabulary.index('EOS')
        self.PAD = self.vocabulary.index('PAD')

        #data_array accepts 1-D number list
        inputs = [[e] for e in data_array]
        targets = [e for e in data_array]

        #the words part
        digits = [self.get_digits(e[0], BASE, MAX_LEN, reverse)[0] for e in inputs]
        lengths = [self.get_digits(e[0], BASE, MAX_LEN, reverse)[1] for e in inputs]
        onehots = [[np.eye(VOCAB_SIZE)[d] for d in e] for e in digits]
        teacher_digits = [[self.SOS] + dlist[:-1] for dlist in digits]
        teacher_onehots = [[np.eye(VOCAB_SIZE)[d] for d in e] for e in teacher_digits]

        self.inputs = torch.tensor(inputs).to(torch.float)
        self.targets = torch.tensor(targets).to(torch.float)
        self.digits = torch.tensor(digits).to(torch.long)
        self.lengths = lengths
        self.onehots = torch.tensor(onehots).to(torch.float)
        self.teacher_digits = torch.tensor(teacher_digits).to(torch.long)
        self.teacher_onehots = torch.tensor(teacher_onehots).to(torch.float)
    
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return {'inputs': self.inputs[idx],
                'targets': self.targets[idx],
                'digits': self.digits[idx],
                'lengths': self.lengths[idx], 
                'onehots': self.onehots[idx],
                'teacher_digits': self.teacher_digits[idx],
                'teacher_onehots': self.teacher_onehots[idx]}

    def get_digits(self, n, base, max_len, reverse):
        digit_list = []
        while n > 0:
            digit_list.insert(0, n%base)
            n = n // base
        #if len(digit_list) == 0:
        #    digit_list.append(0)
        if reverse:
            digit_list.reverse()

        #record sequence lengths
        length = len(digit_list)
        
        #add EOS token (=base)
        if length < max_len:
            digit_list.append(self.EOS)

        #add padding tokens (=base+1)
        while len(digit_list) < max_len:
            digit_list.append(self.PAD)

        return digit_list, length

def dataset_split(n, soft):
    if n == 1000:
        x_test = [e for e in range(n)]
        a = [e for e in range(20)]
        b = random.sample(range(20, 100), 50)
        c = random.sample(range(100,1000), 100)
        x_train = a + b + c
        for e in x_train:
            x_test.remove(e)
        x_valid = random.sample(x_test, 200)
        for e in x_valid:
            x_test.remove(e)

    if n == 100:
        x_train = [e for e in range(n)]
        x_valid = random.sample(range(20,n), 10)
        for e in x_valid:
            x_train.remove(e)
        x_valid += list(range(100,120))
        x_test = random.sample(range(20,n), 10)
        for e in x_test:
            x_train.remove(e)

    if n < 100:
        x_train = [e for e in range(50)]
        x_valid = [e for e in range(50,n)]
        x_test = [e for e in range(100)]

    if n == 10:
        x_train = [e for e in range(10)]
        x_valid = [e for e in range(20)]
        x_test = x_train

    #hard split, n, 300, n 
    if soft == False:
        x_train = list(range(n))
        x_valid = random.sample(range(100,1000),100) + random.sample(range(1000,10000),100) + random.sample(range(10000,100000),100)
        x_test = x_train
    
    if soft == 'veryhard':
        x_train = list(range(200))
        x_valid = random.sample(range(200,1000), 200)
        x_test = list(range(200,1000))
        for e in x_valid:
            x_test.remove(e)
    if soft == 'work':
        x_train = list(range(1000))
        x_valid = list(range(10000))
        x_test = list(range(1000,10000))
        

    #new split, 800/100/100inter/100extra 1000:2000/100extra 2000:10000/100extra 10000:100000/
    if soft == "ultimate":
        x_train = list(range(100,1000))
        x_valid = random.sample(x_train, 100)
        for e in x_valid:
            x_train.remove(e)
        x_test = random.sample(x_train, 100)
        for e in x_test:
            x_train.remove(e)
        x_train += list(range(0,100))

    x_train.sort()
    x_valid.sort()
    x_test.sort()

    return x_train, x_valid, x_test



def make_data_list_identity(n, base, max_len):
    preset = args.preset
    BASE = base
    n = n
    MAX_LEN = max_len


    if preset == 1:
        #preset 1: train(0-99, 80), valid(0-99, 10)(100-999,10)
        n = 100
        x_train = [e for e in range(n)]

        x_valid_inter = random.sample(x_train, n//10)
        for e in x_valid_inter:
            x_train.remove(e)
        x_valid_extra = random.sample(range(n, n*10), n//10)

        x_test_inter = random.sample(x_train, n//10)
        for e in x_test_inter:
            x_train.remove(e)
        x_test_extra = random.sample(range(n, n*10), n//10)

    elif preset == 2:
        #preset 2: train(0-200, 160), valid(0-200, 20)(200-2000,20)
        n = 200
        x_train = [e for e in range(n)]

        x_valid_inter = random.sample(x_train, n//10)
        for e in x_valid_inter:
            x_train.remove(e)
        x_valid_extra = random.sample(range(n, n*10), n//10)

        x_test_inter = random.sample(x_train, n//10)
        for e in x_test_inter:
            x_train.remove(e)
        x_test_extra = random.sample(range(n, n*10), n//10)

    elif preset == 3:
        #preset 2: train(0-200, 160), valid(0-200, 20)(200-20000,20)
        n = 200
        x_train = [e for e in range(n)]

        x_valid_inter = random.sample(x_train, n//10)
        for e in x_valid_inter:
            x_train.remove(e)
        x_valid_extra = random.sample(range(n, n*100), n//10 *2)

        x_test_inter = random.sample(x_train, n//10)
        for e in x_test_inter:
            x_train.remove(e)
        x_test_extra = random.sample(range(n, n*100), n//10 * 2)

    elif preset == 4:
        #preset 2: train(0-2000, 160), valid(0-200, 20)(200-20000,20)
        n = 2000
        x_train = [e for e in range(n)]

        x_valid_inter = random.sample(x_train, n//100)
        for e in x_valid_inter:
            x_train.remove(e)
        x_valid_extra = random.sample(range(n, n*100), n//100)

        x_test_inter = random.sample(x_train, n//10)
        for e in x_test_inter:
            x_train.remove(e)
        x_test_extra = random.sample(range(n, n*100), n//10)

    elif preset == 5:
        #no full range test set, only test on extrapolation
        n = 200
        x_train = [e for e in range(n)]

        x_valid_inter = [e for e in range(n)]
        x_valid_extra = random.sample(range(n, n*100), n//10)

        x_test_inter = [e for e in range(n)]
        x_test_extra = random.sample(range(n, n*100), n//10)
    elif preset >= 100:
        n = preset
        m = 10
        x_train = [e for e in range(n)]
        x_valid_inter=[e for e in range(n)]
        #x_valid_extra = random.sample(range(n, n*m), n)
        x_valid_extra = [e for e in range(n,n+1)]
        x_test_inter = [e for e in range(n)]
        x_test_extra = [e for e in range(n*m)]

    else: 
        n = 30000
        x_train = [e for e in range(200000)]
    
        x_valid_inter = x_train
        x_valid_extra = x_train
        x_test_inter = [1]
        x_test_extra = x_train