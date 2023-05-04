import numpy as np
import torch
from torch import nn
from config import configs
from utils import preprocess
from utils.helper import AverageMeter

def run(model, val_data):

    MSE_criterion = nn.MSELoss()

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    losses = AverageMeter()
    for i in range(len(val_data)):
        unit_data = val_data[i]
        test_dat = torch.Tensor(preprocess.reshape_patch(unit_data, configs.patch_size))
        img_gen = torch.Tensor(model.test(test_dat, real_input_flag))
        loss = MSE_criterion(img_gen, test_dat[:, 1:])
        losses.update(loss)
    return losses

