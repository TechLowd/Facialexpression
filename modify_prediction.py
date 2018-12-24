# -*- coding: utf-8 -*-

import os
import pdb
import numpy as np

if __name__ == "__main__":
    directory = './testtest'
    files = os.listdir(directory)
    for f in files:
        if f.find('sync_blendshape.txt') > 0:
            blendshape = open(os.path.join(directory, f), 'r', encoding='utf-8-sig').readlines()
            truth = open(os.path.join(directory, f.replace('sync_', '')), 'r', encoding='utf-8-sig').readlines()
            modify_blendshape = open(os.path.join(directory, f.replace('sync_', 'm_')), 'w')
            writein = np.zeros((51,))
            for i in range(min(len(truth), len(blendshape))):
                tmp_blend = blendshape[i].strip().split(' ')
                tmp_truth = truth[i].strip().split(' ')
                for j in range(19):
                    writein[j] = float(tmp_blend[j])
                for j in range(19, 51):
                    writein[j] = float(tmp_truth[j])
                modify_blendshape.write(" ".join(writein.astype(str)) + '\n')
            modify_blendshape.close()
