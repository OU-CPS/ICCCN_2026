import torch
from torch.utils.data import Dataset


class Seq2PointDatasetWithTotalTime(Dataset):
    def __init__(self, X, Y, total_center, hour_ids, dow_ids):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.TC = torch.from_numpy(total_center).float()
        self.H = torch.from_numpy(hour_ids).long()
        self.D = torch.from_numpy(dow_ids).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.TC[i], self.H[i], self.D[i]