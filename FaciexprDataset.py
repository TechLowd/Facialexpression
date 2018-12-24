import torch
import torch.utils.data as data

import pdb
import numpy as np

import FaciexprParam as hp

class FaciexprDataset(data.Dataset):
    """头部旋转动作数据集"""

    def __init__(self, directory):
        with open(r'%s' % directory, 'r', encoding='utf-8-sig') as f:
            self.data = f.readlines()
        with open(r'%s' % directory.replace('p_idx', 'r_blendshape'), 'r', encoding='utf-8-sig') as f:
            self.label_r = f.readlines()
        self.length = len(self.data)

    def __getitem__(self, item):
        input_data = self.data[item].strip().split(' ')
        input_data = np.array(input_data).astype('int')
        input_position = np.arange(input_data.shape[0]) + 1  ##

        input_label_r = self.label_r[item].strip().split(' ')
        input_label_r = [elem.split('_') for elem in input_label_r]
        input_label_r = np.array(input_label_r).astype('float')

        data_pad = np.zeros((hp.max_length, ))
        data_pad[:input_data.shape[0]] = input_data
        position_pad = np.zeros((hp.max_length, ))
        position_pad[:input_position.shape[0]] = input_position

        label_r_pad = np.zeros((hp.max_length, input_label_r.shape[1]))
        label_r_pad[:input_label_r.shape[0], :] = input_label_r

        data_pad     = torch.from_numpy(data_pad)
        position_pad = torch.from_numpy(position_pad)
        label_r_pad = torch.from_numpy(label_r_pad)
        ori_length  = torch.tensor(input_data.shape[0])

        return data_pad, position_pad, label_r_pad, ori_length
        
    def __len__(self):
        return self.length
