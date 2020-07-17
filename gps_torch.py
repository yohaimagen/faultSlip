import pandas as pd
from disloc_torch import disloc_pytorch
from faultSlip.disloc import disloc
import numpy as np

import torch


class Gps:
    def __init__(self, data, origin_lon=None, origin_lat=None):
        self.data = pd.read_csv(data)
        self.G_ss = None
        self.G_ds = None
        self.G_o = None
        self.sources_mat = None
#
    def calc_disp(self, model, poisson_ratio=0.25):
        east = torch.DoubleTensor(self.data.x.values * 1e-3)
        north = torch.DoubleTensor(self.data.y.values * 1e-3)
        disp = disloc_pytorch(model[0], model[1], model[2], model[3], model[4], model[5], model[6], 0, 0,
                              model[7], east, north, poisson_ratio)
        return torch.cat(disp)

    def loss(self, model):
        pred_disp = self.calc_disp(model)
        disp = torch.DoubleTensor(np.concatenate((self.data.E.values, self.data.N.values, self.data.Up.values)))
        x = pred_disp.data.numpy()
        y =  disp.numpy()
        return torch.norm(disp - pred_disp) / torch.norm(disp)
