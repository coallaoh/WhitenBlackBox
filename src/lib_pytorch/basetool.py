__author__ = 'joon'

import torch
from util.ios import save_to_cache, load_from_cache


def da(x):
    return x.data.cpu().numpy()


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, listobject):
        self.listobject = listobject

    def __getitem__(self, index):
        return self.listobject[index]

    def __len__(self):
        return len(self.listobject)


class TrainCurve(object):
    def __init__(self, *args):
        self.curves = dict(zip(args, [[] for _ in range(len(args))]))

    def save(self, loc):
        save_to_cache(self.curves, loc)

    def load(self, loc):
        self.curves = load_from_cache(loc)
