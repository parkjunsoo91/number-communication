import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, _i, _j, _k):
        # 3 attributes, 2 instances -> 8 combinations
        self.frame = []
        self._i = _i
        self._j = _j
        self._k = _k
        for i in range(_i):
            for j in range(_j):
                for k in range(_k):
                    i_vec = np.zeros((_i))
                    i_vec[i] = 1
                    j_vec = np.zeros((_j))
                    j_vec[j] = 1
                    k_vec = np.zeros((_k))
                    k_vec[k] = 1
                    
                    label = np.array([i*_j*_k + j*_k + k])
                    #label = np.array([k*_j*_i + j*_i + i])
                    item = np.concatenate((i_vec, j_vec, k_vec, label))
                    self.frame.append(item)
        self.frame = np.array(self.frame)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        sample = {'x': self.frame[idx][0:self._i + self._j + self._k], 'y': self.frame[idx][-1]}        
        return sample
    def get_frame(self):
        x = self.frame[:,0:self._i + self._j + self._k]
        y = self.frame[:,-1]
        return np.array(x), np.array(y)