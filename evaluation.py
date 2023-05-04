import numpy as np
import pandas as pd
import time
from utils import preprocess, metrics
from config import configs
from tqdm import tqdm
from models.model_factory import Model
from utils.dataloader import Dataset


def run(model, test_data):

    pre_result = []
    true_result = []
    print('预测中...')
    test_time = 0
    for i in tqdm(range(len(test_data)), ncols=50):
        unit_data = test_data[i]
        test_dat = preprocess.reshape_patch(unit_data, configs.patch_size)
        test_ims = unit_data[:, :, :, :, :configs.img_channel]
        start_time = time.time()
        img_gen = model.test(test_dat)
        test_time += time.time() - start_time
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_out = img_gen[:, -output_length:]
        trueValue = test_ims[:, configs.input_length:, :, :, :]
        true_result.append(trueValue)
        pre_result.append(img_out)
    print('test_time(avg):', test_time / len(test_data))
    true_np = np.array(true_result)
    pre_np = np.array(pre_result)
    return matrix(true_np, pre_np)

def matrix(true_result, pre_result):
    print('计算指标...')
    for h in range(true_result.shape[3]):
        for w in range(true_result.shape[4]):
            if true_result[:, :, :, h, w, 0].all() == 0:
                pre_result[:, :, :, h, w, 0] = 0

    maes, rmses, accs, r2s = [], [], [], []
    for k in tqdm(range(true_result.shape[2]), ncols=50):
        all_t = []
        all_p = []
        for i in range(true_result.shape[0]):
            for j in range(true_result.shape[1]):
                true = true_result[i, j, k, :, :, 0]
                pred = pre_result[i, j, k, :, :, 0]
                true = list(true.reshape(true.shape[0] * true.shape[1]))
                pred = list(pred.reshape(pred.shape[0] * pred.shape[1]))
                all_t.extend(true)
                all_p.extend(pred)
        mae = round(metrics.mae_rmse(all_t, all_p)[0],3)
        rmse = round(metrics.mae_rmse(all_t, all_p)[1],3)
        acc = round(metrics.acc(all_t, all_p), 3)
        r2 = round(metrics.r2(all_t, all_p),3)
        maes.append(mae)
        rmses.append(rmse)
        accs.append(acc)
        r2s.append(r2)
        print(k, mae, rmse, acc, r2)
    target = pd.DataFrame()
    target.insert(0,'mae',maes)
    target.insert(1, 'rmse', rmses)
    target.insert(2, 'acc', accs)
    target.insert(3, 'r2', r2s)
    target.to_csv('performance.csv', index=0)


if __name__ == '__main__':
    dataset = Dataset((configs.train_p, 1 - configs.train_p))
    train_batchs, test_batchs = dataset.get_sets()
    print('model loading...')
    model = Model(configs)
    model.load(configs.pretrained_model)
    run(model, test_batchs)
