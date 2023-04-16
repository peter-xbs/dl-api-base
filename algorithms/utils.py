# _*_ coding:utf-8 _*_
# @Time: 2023-04-15 21:53
# @Author: cmcc
# @Email: xinbao.sun@hotmail.com
# @File: utils.py
# @Project: myabtest

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len):
        self.data = [item[:max_len] for item in data]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)