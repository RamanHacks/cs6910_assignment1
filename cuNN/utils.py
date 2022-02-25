"""Code is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py
"""
import cupy as cp
import numpy as np
import pickle
from PIL import Image, ImageOps, ImageEnhance


def cosine_decay_with_warmup(global_step,
                                learning_rate_base,
                                total_steps,
                                warmup_learning_rate=0.00001,
                                warmup_steps=0,
                                hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in
        Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
        ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
        a float representing learning rate.
    Raises:
        ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                            'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (
        1 + np.cos(
            np.pi * (global_step - warmup_steps - hold_base_rate_steps) /
            float(total_steps - warmup_steps - hold_base_rate_steps)
            )
        )
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                    learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                                'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                    learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


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


##############################################################################



def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    return image - 127

def apply_op(image, op, severity):
    #   image = np.clip(image, 0, 255)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img)

def augment_and_mix(image, severity=3, width=3, depth=-1, beta=.2, prob=1.):
    image = image[:, :, np.newaxis]
    image = np.repeat(image, 3, -1) 
    if np.random.rand(1) < prob:
        ws = np.float32(
        np.random.dirichlet([beta] * width))
        m = np.float32(np.random.beta(beta, beta))
    else:
        return image
    
    mix = np.zeros_like(image).astype(np.float32)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug

    mixed = (1 - m) * image + m * mix
    return mixed

def augmix_batch(image_list, severity=5, width=5, depth=-1, beta=2, prob=1.):
    from joblib import Parallel, delayed
    batch_out = Parallel(n_jobs=12)(delayed(augment_and_mix)(i,severity,width,depth,beta,prob) for i in image_list)
    return batch_out
