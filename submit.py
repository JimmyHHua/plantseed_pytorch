#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-01-24 18:07:47

@author: JimmyHua
"""
import argparse
import logging
import os
import sys

import pandas as pd
import torch
from torch import nn
from torch.backends import cudnn

import network
from core.loader import get_test_set

logging.basicConfig(
    level = logging.INFO, #打印日志级别数值
    format = '%(asctime)s: %(message)s', #输出时间和信息
    stream=sys.stdout #指定日志的输出流
    )

def submit(args):
    test_loader, id_to_class = get_test_set(args.bs)
    net = getattr(network, 'resnet50')(classes=12)
    net.load_state_dict(torch.load(args.model_path)['state_dict'])
    net = nn.DataParallel(net,device_ids=[0,1])
    net.eval()
    net = net.cuda()

    pred_labels = list()
    indices = list()
    print('model has been loaded!')
    for data, fname in test_loader:
        data = data.cuda()
        with torch.no_grad():
            scores = net(data)
        labels = scores.max(1)[1].cpu().numpy().tolist()
        pred_labels.extend(labels)
        indices.extend(fname)
    df = pd.DataFrame({'file': indices, 'species': pred_labels})
    df['species'] = df['species'].apply(lambda x: id_to_class[x])
    df.to_csv('submission.csv', index=False)
    print('Finished!')


def main():
    parser = argparse.ArgumentParser(description='plantseed model testing')
    parser.add_argument('--model_path', type=str, default='checkpoints/model_best.pt',
                        help='training batch size')
    parser.add_argument('--bs', type=int, default=128, help='testing batch size')

    args = parser.parse_args()
    cudnn.benchmark = True
    submit(args)


if __name__ == '__main__':
    main()