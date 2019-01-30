#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-01-29 17:26:46

@author: JimmyHua
"""
import argparse
import logging
import sys

import torch
from torch.backends import cudnn
import network
from core.loader import get_train_set
from core.train_engine import Train_Engine
from tools.lr_scheduler import LRScheduler
#import torchvision.models as models

logging.basicConfig(
    level = logging.INFO, #打印日志级别数值
    format = '%(asctime)s: %(message)s', #输出时间和信息
    stream=sys.stdout #指定日志的输出流
    )

def train(args):
    logging.info('=========== Starting Training ============')
    train_data, valid_data = get_train_set(args.bs)
    net = getattr(network, 'resnet50')(classes=12)

    optimizer = torch.optim.SGD(net.parameters(),lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = LRScheduler(base_lr=args.lr, step=args.step, factor=args.factor,
                               warmup_epoch=args.warmup_epoch, warmup_begin_lr=args.warmup_begin_lr)

    net = torch.nn.DataParallel(net,device_ids=[0,1])
    net = net.cuda()
    model = Train_Engine(net)
    model.fit(train_data=train_data, test_data=valid_data, optimizer=optimizer, criterion=criterion, lr_scheduler=lr_scheduler,
            epochs=args.epochs, print_interval=args.print_interval, eval_step=args.eval_step,save_step=args.save_step, save_dir=args.save_dir)
def main():
    parser = argparse.ArgumentParser(description='plantseed trainning')
    parser.add_argument('--bs',type=int, default=128, help='batch_size')
    parser.add_argument('--lr',type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd',type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--factor',type=float, default=0.1, help='factor')
    parser.add_argument('--warmup_begin_lr',type=float, default=1e-4, help='warmup_begin_lr')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--epochs', type=int, default=600, help='training epochs')
    parser.add_argument('--warmup_epoch', type=int, default=5, help='training warmup_epoch')
    parser.add_argument('--step', type=list, default=[20,30], help='training step')
    parser.add_argument('--print_interval', type=int, default=10, help='how many iterations to print')
    parser.add_argument('--eval_step', type=int, default=1, help='how many epochs to evaluate')
    parser.add_argument('--save_step', type=int, default=10, help='how many epochs to save model')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='save model directory')

    args=parser.parse_args()
    cudnn.benchmark = True
    train(args)

if __name__ == '__main__':
    main()