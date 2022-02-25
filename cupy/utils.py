"""Code is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py
"""
import cupy as cp
import numpy as np
import pickle

def save(file, name):
    with open(name, 'wb') as f:
        pickle.dump(file, f)

def load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
   
def smooth_batch(target, smoothing=0.1, num_classes=10):
    target *= (1. - smoothing)
    target += (smoothing / num_classes)
    return target


def mix_target(target_a, target_b, lam, num_classes=10, smoothing=0.):
    y1 = smooth_batch(target_a, smoothing, num_classes)
    y2 = smooth_batch(target_b, smoothing, num_classes)
    return lam * y1 + (1. - lam) * y2 

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = cp.sqrt(1. - lam)
    cut_w = cp.int(W * cut_rat)
    cut_h = cp.int(H * cut_rat)

    # uniform
    cx = cp.random.randint(W)
    cy = cp.random.randint(H)

    bbx1 = cp.clip(cx - cut_w // 2, 0, W)
    bby1 = cp.clip(cy - cut_h // 2, 0, H)
    bbx2 = cp.clip(cx + cut_w // 2, 0, W)
    bby2 = cp.clip(cy + cut_h // 2, 0, H)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (size[1] * size[2]))

    return bbx1, bby1, bbx2, bby2, lam


def cutmix_batch(image, target, prob, beta=.2, num_classes=10):
    if cp.random.rand(1) < prob:
        lam = cp.random.beta(beta, beta)
    else:
        return image, target

    rand_index = cp.random.permutation(image.shape[0])
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2, lam = rand_bbox(image.shape, lam)
    image[:, bbx1:bbx2, bby1:bby2] = image[rand_index, bbx1:bbx2,
                                              bby1:bby2]
    target = mix_target(target_a,target_b,lam)
    return image, target


def mixup_batch(image, target, prob, beta=.2, num_classes=10):
    if cp.random.rand(1) < prob:
        lam = cp.random.beta(beta, beta)
    else:
        return image, target

    rand_index = cp.random.permutation(image.shape[0])
    target_a = target
    target_b = target[rand_index]
    
    image_a = image
    image_b = image[rand_index]
    
    image = mix_target(image_a,image_b,lam)
    target = mix_target(target_a,target_b,lam)
    return image, target