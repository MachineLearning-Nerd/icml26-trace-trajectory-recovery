import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import ipdb as pdb

class CartpoleDataset(Dataset):

    def __init__(self, directory, dataset='cartpole', transform='default'):
        super().__init__()
        self.path = os.path.join(directory, dataset)
        self.domain_names = ['v0', 'v1', 'v2', 'v3', 'v4']
        self.num_domains = 5
        self.datums_per_domain = 1000
        self.samples_per_datum = 40
        self.length = 3

    def __len__(self):
        length = self.num_domains * self.datums_per_domain * (self.samples_per_datum - self.length + 1)
        return length
    
    def __getitem__(self, idx):
        offset = self.samples_per_datum - self.length + 1
        src_domain = idx % self.num_domains
        src_rollout = (idx // self.num_domains) // offset
        src_timestep = (idx // self.num_domains) % offset
        data_path = os.path.join(self.path, 
                                 str(self.domain_names[src_domain]), 
                                 'trail_%d.npz'%src_rollout)
        datum = np.load(data_path)
        frames = datum['obs'][src_timestep:src_timestep+self.length]
        frames = frames.reshape(self.length, 1, 128, 128)
        # x, x_dot, theta, theta_dot = state
        states = datum['state'][src_timestep:src_timestep+self.length]
        
        # 去掉action，只返回必要的数据
        sample = {
            'xt': frames.astype('float32'), 
            'yt': states.astype('float32'), 
            'ct': src_domain
        }

        return sample


class CartpoleDatasetTwoSample(Dataset):

    def __init__(self, directory, dataset='cartpole', transform='default'):
        super().__init__()
        self.path = os.path.join(directory, dataset)
        self.domain_names = ['v0', 'v1', 'v2', 'v3', 'v4']
        self.num_domains = 5
        self.datums_per_domain = 1000
        self.samples_per_datum = 40
        self.length = 3

    def __len__(self):
        length = self.num_domains * self.datums_per_domain * (self.samples_per_datum - self.length + 1)
        return length
    
    def retrieve_by_index(self, idx):
        offset = self.samples_per_datum - self.length + 1
        src_domain = idx % self.num_domains
        src_rollout = (idx // self.num_domains) // offset
        src_timestep = (idx // self.num_domains) % offset
        data_path = os.path.join(self.path, 
                                 str(self.domain_names[src_domain]), 
                                 'trail_%d.npz'%src_rollout)
        datum = np.load(data_path)
        frames = datum['obs'][src_timestep:src_timestep+self.length]
        frames = frames.reshape(self.length, 1, 128, 128)
        # x, x_dot, theta, theta_dot = state
        states = datum['state'][src_timestep:src_timestep+self.length]
        return frames.astype('float32'), states.astype('float32'), src_domain

    def __getitem__(self, idx):
        xt, yt, ct = self.retrieve_by_index(idx)
        idx_rnd = random.randint(0, self.__len__()-1)
        xtr, ytr, ctr = self.retrieve_by_index(idx_rnd)

        sample = {
            "s1": {"yt": yt, "xt": xt, "ct": ct},
            "s2": {"yt": ytr, "xt": xtr, "ct": ctr}
        }

        return sample