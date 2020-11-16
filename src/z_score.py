import math
import numpy as np

TOTAL_NUM = 10000

class Z_Score():
    def __init__(self):
        self.Rs = []
    
    def GetZ(self, reward):
        self.Rs.append(reward)
        if len(self.Rs) > TOTAL_NUM:
            self.Rs.pop(0)

        _avg = np.mean(self.Rs)
        _std = np.std(self.Rs)

        _z = (reward - _avg) / (_std + 1e-6)
        return _z
    
    def normalize(self, r_batch):
        for r in r_batch:
            self.Rs.append(r)
            if len(self.Rs) > TOTAL_NUM:
                self.Rs.pop(0)

        ret = []
        _avg = np.mean(self.Rs)
        _std = np.std(self.Rs)
        for r in r_batch:
            _z = (r - _avg) / (_std + 1e-6)
            ret.append(_z)
        return np.vstack(ret)
