#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-01-29 14:12:22

@author: JimmyHua
"""
import math
import random


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.p = p
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self,img):
        if random.uniform(0,1)>self.p:
            return img

        for _ in range(100):
            area = img.size()[1]*img.size()[2]

            target_area = random.uniform(self.sl, self.sh)*area
            target_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * target_ratio)))
            w = int(round(math.sqrt(target_area / target_ratio)))

            if h < img.size()[1] and w < img.size()[2]:
                x1 = random.randint(0, img.size()[1]-h)
                y1 = random.randint(0, img.size()[2]-w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]

                return img
                
        return img
