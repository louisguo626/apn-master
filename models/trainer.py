import os.path
import datetime
import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as sk_cpt_ssim
from utils import preprocess, metrics


def train(model, ims, configs, itr=None):  #itr->epoches
    cost = model.train(ims)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()               #颠倒时序再训练
        cost += model.train(ims_rev)
        cost = cost / 2
    return cost
