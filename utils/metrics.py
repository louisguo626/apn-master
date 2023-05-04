import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def acc(true, pred):
    up, down = 0, 0
    for i in range(len(true)):
        up += abs(pred[i] - true[i])
        down += true[i]
    return 1 - (up / down)


def mae_rmse(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    return mae, rmse


def r2(true, pred):
    return r2_score(true, pred)

def corr(x, y):
    up = 0
    down1 = 0
    down2 = 0
    for i in range(len(x)):
        up += x[i] * y[i]
        down1 += x[i] * x[i]
        down2 += y[i] * y[i]
    return up / (np.sqrt(down1 * down2))


if __name__ == '__main__':
    true_result = np.load('../results/true100.npy')
    pre_result = np.load('../results/convlstm_pre.npy')
    accs = []
    for i in tqdm(range(true_result.shape[0]), ncols=50):
        for j in range(true_result.shape[1]):
            true = true_result[i, j, 0, :, :, 0]
            pred = pre_result[i, j, 0, :, :, 0]
            true = true.reshape(true.shape[0] * true.shape[1])
            pred = pred.reshape(pred.shape[0] * pred.shape[1])
            accs.append(acc(true, pred))
    accs = np.array(accs)
    print(accs.mean())
