# coding=utf-8
# tensorboard --logdir F:\PycharmCode\TPDNet\src\out

import sys
import datetime
import random
import numpy as np
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import src.dataset as dataset
from src.net import CTDNet
from apex import amp


def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)

    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter+1)/(union-inter+1)
    iou = iou.mean()
    return iou + 0.6*bce


def bce_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)
    return bce


def validate(model, val_loader, nums):
    model.train(False)
    avg_mae = 0.0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out, _, _, _, _, _ = model(image)
            pred = torch.sigmoid(out[0, 0])
            avg_mae += torch.abs(pred - mask[0]).mean()

    model.train(True)
    return (avg_mae / nums).item()


def train(Dataset, Network):
    ## Set random seeds
    seed = 7
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    ## dataset
    cfg = Dataset.Config(datapath='../data/DUTS', savepath='./out', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=48)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True)
    ## val dataloader
    val_cfg = Dataset.Config(datapath='../data/ECSSD', mode='test')
    val_data = Dataset.Data(val_cfg)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    min_mae = 1.0
    best_epoch = 0
    ## network
    net = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    enc_params, dec_params = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            enc_params.append(param)
        else:
            dec_params.append(param)

    optimizer = torch.optim.SGD([{'params': enc_params}, {'params': dec_params}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw = SummaryWriter(cfg.savepath)
    global_step = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
        '''
        if epoch < 40:
            optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (40 + 1) * 2 - 1)) * cfg.lr * 0.1
            optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (40 + 1) * 2 - 1)) * cfg.lr
        else:
            if epoch % 2 == 0:
                optimizer.param_groups[0]['lr'] = (1 - abs((38 + 1) / (40 + 1) * 2 - 1)) * cfg.lr * 0.1
                optimizer.param_groups[1]['lr'] = (1 - abs((38 + 1) / (40 + 1) * 2 - 1)) * cfg.lr
            else:
                optimizer.param_groups[0]['lr'] = (1 - abs((39 + 1) / (40 + 1) * 2 - 1)) * cfg.lr * 0.1
                optimizer.param_groups[1]['lr'] = (1 - abs((39 + 1) / (40 + 1) * 2 - 1)) * cfg.lr
        '''

        for step, (image, mask, edge) in enumerate(loader):
            image, mask, edge = image.cuda().float(), mask.cuda().float(), edge.float().cuda()

            out1, out_edge, out2, out3, out4, out5 = net(image)
            loss1 = total_loss(out1, mask)
            loss_edge = bce_loss(out_edge, edge)
            loss2 = total_loss(out2, mask)
            loss3 = total_loss(out3, mask)
            loss4 = total_loss(out4, mask)
            loss5 = total_loss(out5, mask)
            loss = loss1 + loss_edge + loss2/2 + loss3/4 + loss4/8 + loss5/16

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss1': loss1.item(), 'loss_edge': loss_edge.item(), 'loss2': loss2.item(),
                                    'loss3': loss3.item(), 'loss4': loss4.item(), 'loss5': loss5.item()}, global_step=global_step)
            if step % 10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f' % (datetime.datetime.now(), global_step, epoch+1,
                                                cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))

        if epoch > cfg.epoch/3*2:
            mae = validate(net, val_loader, 1000)
            print('ECSSD MAE:%s' % mae)
            if mae < min_mae:
                min_mae = mae
                best_epoch = epoch + 1
                torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
            print('best epoch is:%d, MAE:%s' % (best_epoch, min_mae))
            if epoch == 46 or epoch == 47:
                torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))


if __name__ == '__main__':
    train(dataset, CTDNet)