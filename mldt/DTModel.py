"""
Created on September 05, 2023

@author: Hassan Sartaj
@version: 1.0
"""

import torch
import learn2learn as l2l
from torch.utils.data import Dataset
from torch import nn
import pandas as pd

devices_names = ["D1", "D2", "D3"]
file_separator = ";"


# -------------------
# Model Architecture
# -------------------
class DTModel(nn.Module):
    def __init__(self, infeatures=19, outfeatures=1, dim=32):
        super(DTModel, self).__init__()
        self.linear1 = nn.Linear(infeatures, dim).to(torch.float64)
        self.linear2 = nn.Linear(dim, outfeatures).to(torch.float64)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


# ---------------------------------
# Meta Dataset & Taskset Creation
# ---------------------------------

feature_cols = None
label_outs = None
status_codes_map = {}


class DeviceDataset(Dataset):

    def __init__(self, file_name=None, device_name=None, i_features=None):
        df = pd.read_csv(file_name, sep=file_separator, skipinitialspace=True)
        df.iloc[:, 0:feature_cols - 1].astype(float)
        x = df.iloc[:, 0:feature_cols - 1].values
        y = df.iloc[:, feature_cols - 1].values

        # if feature_cols < i_features:
        #     diff = i_features - feature_cols
        #     df1 = pd.DataFrame(x)
        #     for d in range(diff + 1):
        #         df1["x" + str(d)] = 1
        #     x = df1.values

        self.x_train = torch.tensor(x, dtype=torch.float64)
        self.y_train = torch.tensor(y, dtype=torch.float64)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


def create_data_task_sets(file_name, ways=1, shots=1, num_tasks=1, i_features=1):
    for dev_n in devices_names:
        if dev_n in file_name:
            device_name = dev_n
            break
    deviceDs = DeviceDataset(file_name, device_name, i_features)
    dataset = l2l.data.MetaDataset(deviceDs)

    transforms = [  # Easy to define your own transform
        l2l.data.transforms.NWays(dataset, n=ways),
        l2l.data.transforms.KShots(dataset, k=2 * shots),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
    ]

    # create taskset
    taskset = l2l.data.Taskset(dataset, transforms, num_tasks=num_tasks)

    # print("taskset: ")
    # for task in taskset.dataset:
    #     X, y = task
    #     print("X: ", X, " - y: ", y)
    #     break

    return taskset
