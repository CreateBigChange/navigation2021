import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch.utils.data as Data
import torch
from torch import nn


def get_pos(data, ref):
    D2R = math.pi / 180;
    Rm = 6351887.65;
    Rn = 6383651.96;
    h = 1115;
    zero_pos = np.array([51.081140, -114.130222]);

    ref_len = ref.shape[0]
    data_len = data.shape[0]
    pos = np.array([])
    # deal with accel ios data
    for k in range(data_len):

        if data.iloc[k, 0] <= ref.iloc[0, 0]:

            if k == 0:

                pos = np.array([ref.iloc[0, 1], ref.iloc[0, 2]])
            else:
                pos = np.vstack((pos, np.array([ref.iloc[0, 1], ref.iloc[0, 2]])))

        elif data.iloc[k, 0] >= ref.iloc[ref_len - 1, 0]:
            if k == 0:
                pos = np.array([ref.iloc[ref_len - 1, 1], ref.iloc[ref_len - 1, 2]])
            else:
                pos = np.vstack((pos, np.array([ref.iloc[ref_len - 1, 1], ref.iloc[ref_len - 1, 2]])))
        else:
            for i in range(ref_len):
                if data.iloc[k, 0] >= ref.iloc[i, 0] and data.iloc[k, 0] <= ref.iloc[i + 1, 0]:

                    dt = data.iloc[k, 0] - ref.iloc[i, 0]
                    head = ref.iloc[i, 3]
                    velocity = ref.iloc[i, 4]
                    d_N = dt * velocity * math.cos(head)
                    d_E = dt * velocity * math.sin(head)

                    dN = d_N / (Rm + h)
                    dE = d_E / ((Rn + h) * math.cos(zero_pos[0]))

                    if k == 0:
                        pos = np.array([ref.iloc[i, 1], ref.iloc[i, 2]] + [dN, dE])
                    else:
                        pos = np.vstack((pos, np.array([ref.iloc[i, 1] + dN, ref.iloc[i, 2]] + dE)))
                    break;

    data['lat'] = pos[:, 0]
    data['lon'] = pos[:, 1]
    return data