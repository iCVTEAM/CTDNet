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
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import src.dataset as dataset
from src.net import CTDNet


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg = Dataset.Config(datapath=path, snapshot='./out/model-48', mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def show(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                out = self.net(image)

                plt.subplot(221)
                plt.imshow(np.uint8(image[0].permute(1,2,0).cpu().numpy()*self.cfg.std + self.cfg.mean))
                plt.subplot(222)
                plt.imshow(mask[0].cpu().numpy())
                plt.subplot(223)
                plt.imshow(out[0, 0].cpu().numpy())
                plt.subplot(224)
                plt.imshow(torch.sigmoid(out[0, 0]).cpu().numpy())
                plt.show()
                input()
    
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
                # pred_12 = (torch.sigmoid(out_12[0, 0]) * 255).cpu().numpy()
                # pred_1 = (torch.sigmoid(out_1[0, 0]) * 255).cpu().numpy()

                save_path = '../eval/maps/CTDNet/' + self.cfg.datapath.split('/')[-1]
                save_edge = '../eval/maps/CTDNet/Edge/' + self.cfg.datapath.split('/')[-1]
                # save_12 = '../eval/maps/CTDNet/Pred_12/' + self.cfg.datapath.split('/')[-1]
                # save_1 = '../eval/maps/CTDNet/Pred_1/' + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not os.path.exists(save_edge):
                    os.makedirs(save_edge)
                '''
                if not os.path.exists(save_12):
                    os.makedirs(save_12)
                if not os.path.exists(save_1):
                    os.makedirs(save_1)
                '''
                cv2.imwrite(save_path+'/'+name[0]+'.png', np.round(pred))
                cv2.imwrite(save_edge + '/' + name[0] + '_edge.png', np.round(edge))
                # cv2.imwrite(save_12 + '/' + name[0] + '.png', np.round(pred_12))
                # cv2.imwrite(save_1 + '/' + name[0] + '.png', np.round(pred_1))

            cost_time.pop(0)
            print('Mean running time is: ', np.mean(cost_time))
            print("FPS is: ", len(self.loader.dataset) / np.sum(cost_time))


if __name__ == '__main__':
    for path in ['../data/ECSSD', '../data/PASCAL-S', '../data/DUTS', '../data/DUT-OMRON', '../data/SOD']:
        test = Test(dataset, CTDNet, path)
        test.save()
        # test.show()   '../data/HKU-IS'
