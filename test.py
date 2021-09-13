# coding=utf-8
# maxF, fm, MAE, wfm, Sm, Em
# matlab -nosplash -nodesktop -r main

import os
import time
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset as dataset
from net import CTDNet


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg = Dataset.Config(datapath=path, snapshot='./out/model-40', mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
    
    def save(self):
        with torch.no_grad():
            cost_time = list()
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                out, out_edge = self.net(image, shape)
                torch.cuda.synchronize()
                cost_time.append(time.perf_counter() - start_time)
                pred = (torch.sigmoid(out[0, 0]) * 255).cpu().numpy()
                edge = (torch.sigmoid(out_edge[0, 0]) * 255).cpu().numpy()

                save_path = '../eval/maps/CTDNet/' + self.cfg.datapath.split('/')[-1]
                save_edge = '../eval/maps/CTDNet/Edge/' + self.cfg.datapath.split('/')[-1]

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not os.path.exists(save_edge):
                    os.makedirs(save_edge)

                cv2.imwrite(save_path+'/'+name[0]+'.png', np.round(pred))
                cv2.imwrite(save_edge + '/' + name[0] + '_edge.png', np.round(edge))

            cost_time.pop(0)
            print('Mean running time is: ', np.mean(cost_time))
            print("FPS is: ", len(self.loader.dataset) / np.sum(cost_time))


if __name__ == '__main__':
    for path in ['../data/ECSSD', '../data/PASCAL-S', '../data/DUTS', '../data/DUT-OMRON', '../data/SOD']:
        test = Test(dataset, CTDNet, path)
        test.save()
