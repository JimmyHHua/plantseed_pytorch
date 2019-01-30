#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-01-29 10:21:34

@author: JimmyHua
"""
import os
import shutil
import numpy as np

def make_dir(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

train_image_path = 'data/train'
train_label = os.listdir(train_image_path)
num_per_label = []

for i in train_label:
    num_per_label.append(len(os.listdir(os.path.join(train_image_path,i))))
num_valid = int(min(num_per_label)*0.2) # trian:valid =5:1

for i in train_label:
    idx_valid = np.random.choice(os.listdir(os.path.join(train_image_path,i)), num_valid, replace=False)
    make_dir(['data','valid',i])
    for img in idx_valid:
        shutil.move(os.path.join('data/train',i,img),os.path.join('data/valid',i,img))
print('Finished!')
