# -*- coding: utf-8 -*-
# @输入一系列{index}_{id}_p_idx.txt文件，一个文件内包含一句话，每个数字表示一个汉字，
# @输出对应的{index}_{id}_sync_skeleton.txt文件，文件内每一行是一帧，每一行的四个数字表示的是四元数。

import os
import cv2
import pdb
import math
import torch
import numpy as np

import FaciexprParam as hp
from FaciexprNet import FaciexprNet

if __name__ == '__main__':
    input_output_directory = './testtest'
    p_idx_files = os.listdir(input_output_directory)
    loadmodel_directory = './checkpoint/1221_r0.5_smooth0.5_withoutpe/epoch_455batch_20020_lossr_0.031241.pkl'
    # loadmodel_directory = './checkpoint/1221_r0.5_smooth0.5_withoutpe/epoch_509batch_22370_lossr_0.039036.pkl'
    # loadmodel_directory = './checkpoint/1221_r0.5_smooth0.5_withpe/epoch_455batch_20020_lossr_0.037324.pkl'
    use_gpu = True
    model = FaciexprNet()
    if use_gpu:
        model = model.cuda()
    model.load_state_dict(torch.load(loadmodel_directory))
    for p_idx_file in p_idx_files:
        if p_idx_file.find('_p_idx.txt') > 0:
            p_idx = open(os.path.join(input_output_directory, p_idx_file), 'r', encoding='utf-8-sig').readlines()[0]
            p_idx = p_idx.strip().split(' ')
            p_idx = np.array(p_idx).astype('int')
            position = np.arange(p_idx.shape[0]) + 1
            ori_length = torch.tensor(p_idx.shape[0])
            p_idx, position = torch.from_numpy(p_idx), torch.from_numpy(position)
            p_idx, position = torch.unsqueeze(p_idx, 0), torch.unsqueeze(position, 0)
            p_idx, position, ori_length = p_idx.long(), position.long(), ori_length.long()
            if use_gpu:
                p_idx, position, ori_length = p_idx.cuda(), position.cuda(), ori_length.cuda()
            
            # out_r = model.forward(p_idx, position)
            out_r = model.forward(p_idx)
            pred = out_r[0, :ori_length, :]
            sync_skeleton_f = open(os.path.join(input_output_directory, p_idx_file).replace('p_idx', 'sync_blendshape'), 'w')
            writein = np.zeros((51,))
            for l in range(ori_length-1):
                writein[0]  = writein[1]  = pred[l][0] * 100
                writein[2]  = writein[3]  = pred[l][1] * 100
                writein[8]  = writein[9]  = pred[l][2] * 100
                writein[14] = writein[15] = pred[l][3] * 100
                writein[16] = pred[l][4] * 100
                writein[17] = writein[18] = pred[l][5] * 100
                sync_skeleton_f.write(' '.join(writein.astype(str)) + '\n')
            sync_skeleton_f.close()
