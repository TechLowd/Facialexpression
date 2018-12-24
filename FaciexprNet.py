# -*- coding: utf-8 -*-

import os
import pdb
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import FaciexprParam as hp
from FaciexprDataset import FaciexprDataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
parser = argparse.ArgumentParser(description='Face Pytorch implementation.')

parser.add_argument('--use_gpu',            type=bool,  help='whether to use gpu',                      default=True)
parser.add_argument('--mode',               type=str,   help='train or test',                           default='train')
parser.add_argument('--data_path',          type=str,   help='path to the data',                        default='./data/')
parser.add_argument('--log_directory',      type=str,   help='directory to save log summaries',         default='./log/1221_r0.5_smooth0.5/')
parser.add_argument('--checkpoint_path',    type=str,   help='path to a specific checkpoint to load',   default='./checkpoint/1221_r0.5_smooth0.5/')
parser.add_argument('--batch_size',         type=int,   help='batch size',                              default=64)
parser.add_argument('--num_epochs',         type=int,   help='number of epochs',                        default=512)
parser.add_argument('--num_gpus',           type=int,   help='number of GPUs to use for training',      default=1)
parser.add_argument('--learning_rate',      type=float, help='initial learning rate',                   default=1e-4)

args = parser.parse_args()

train_dataset = FaciexprDataset(directory=os.path.join(args.data_path, 'p_idx.txt'))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

class FaciexprNet(nn.Module):
    
    def __init__(self):
        super(FaciexprNet, self).__init__()

        self.use_gpu = args.use_gpu
        self.embed_data = nn.Embedding(hp.data_dim, hp.embed_dim)
        self.embed_position = nn.Embedding(hp.position_dim, hp.embed_dim)
        self.conv1 = nn.Conv1d(in_channels=hp.embed_dim, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=hp.embed_dim, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=hp.embed_dim, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(in_channels=hp.embed_dim, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.regress = nn.Sequential(
            nn.Linear(hp.hidden_dim, 6),
            nn.Sigmoid()
        )
        self.loss_func_r = nn.L1Loss(reduce=False)
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

    def gradient(self, out):
        gx = out[:, :-1] - out[:, 1:]
        return torch.abs(gx)
    
    # def forward(self, data, position):
    def forward(self, data):
        # out = self.embed_data(data) + self.embed_position(position)
        out = self.embed_data(data)
        out = torch.transpose(out, 1, 2)
        
        tmp = self.conv1(out)
        tmp_a, tmp_b = torch.split(tmp, hp.embed_dim, dim=1)
        out = out + tmp_a * torch.sigmoid(tmp_b)
        
        tmp = self.conv2(out)
        tmp_a, tmp_b = torch.split(tmp, hp.embed_dim, dim=1)
        out = out + tmp_a * torch.sigmoid(tmp_b)
        
        tmp = self.conv3(out)
        tmp_a, tmp_b = torch.split(tmp, hp.embed_dim, dim=1)
        out = out + tmp_a * torch.sigmoid(tmp_b)
        
        tmp = self.conv4(out)
        tmp_a, tmp_b = torch.split(tmp, hp.embed_dim, dim=1)
        out = out + tmp_a * torch.sigmoid(tmp_b)

        out = torch.transpose(out, 1, 2)  # (N, 550, 6)
        pred_r = self.regress(out)
        return pred_r
    
    def _train(self, epoch):
        self.train()

        for i, train_data in enumerate(train_loader, 1):
            batch_num = len(train_loader) * (epoch - 1) + i
            
            data, position, label_r, ori_length_lst = train_data
            data, position, label_r = data.long(), position.long(), label_r.float()
            if self.use_gpu:
                data, position, label_r, ori_length_lst = data.cuda(), position.cuda(), label_r.cuda(), ori_length_lst.cuda()
            # out_r = self.forward(data, position)
            out_r = self.forward(data)

            curr_batch_size = len(ori_length_lst) 

            mask_mat = torch.ones(curr_batch_size, hp.max_length)
            total_num = torch.tensor(0)
            if self.use_gpu:
                mask_mat = mask_mat.cuda()
                total_num = total_num.cuda()
            for j in range(curr_batch_size):
                mask_mat[j, ori_length_lst[j]:] = 0
                total_num = total_num + ori_length_lst[j]
            
            loss_mat_r = [self.loss_func_r(out_r[:, :, j], label_r[:, :, j]) for j in range(6)]
            loss_r = torch.mean(mask_mat * loss_mat_r[0]) + torch.mean(mask_mat * loss_mat_r[1]) + torch.mean(mask_mat * loss_mat_r[2])  + \
                     torch.mean(mask_mat * loss_mat_r[3]) + torch.mean(mask_mat * loss_mat_r[4]) + torch.mean(mask_mat * loss_mat_r[5])
            smoothness_loss_mat = [self.gradient(out_r[:, :, j]) for j in range(6)]
            smoothness_loss = torch.mean(mask_mat[:,:-1] * smoothness_loss_mat[0]) + torch.mean(mask_mat[:,:-1] * smoothness_loss_mat[1]) + \
                              torch.mean(mask_mat[:,:-1] * smoothness_loss_mat[2]) + torch.mean(mask_mat[:,:-1] * smoothness_loss_mat[3]) + \
                              torch.mean(mask_mat[:,:-1] * smoothness_loss_mat[4]) + torch.mean(mask_mat[:,:-1] * smoothness_loss_mat[5])
            loss = 0.5 * loss_r + 0.5 * smoothness_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('epoch: {}, batch: {}, Loss_r: {:.6f}'.format(epoch, batch_num, loss_r))
            writer.add_scalar('Train/Loss_all', loss, batch_num)
            writer.add_scalar('Train/Loss_r', loss_r, batch_num)

            if batch_num % 10 == 0 and batch_num > 20000:
                torch.save(self.state_dict(), os.path.join(args.checkpoint_path, 'epoch_{}batch_{}_lossr_{:.6f}.pkl'.format(epoch, batch_num, loss_r)))

if __name__ == '__main__':
    writer = SummaryWriter(log_dir=args.log_directory)
    model = FaciexprNet()
    if args.use_gpu:
        model = model.cuda()
    # loadmodel_directory = 'E:/face_r/checkpoint/124_r_c_smooth/epoch_508batch_19800_acc_92.02_loss_0.035612.pkl'
    # model.load_state_dict(torch.load(loadmodel_directory))
    for epoch in range(1, args.num_epochs + 1):
        model._train(epoch=epoch)
