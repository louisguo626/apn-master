import numpy as np
import random
from config import configs


class Dataset:
    def __init__(self, percent):
        super().__init__()
        self.train_p = percent[0]
        self.val_p = percent[0]
        self.test_p = percent[1]
        self.seed = configs.seed

    def split_list(self, data):  # Sliding window slice
        data = np.array(data)
        rows = data.shape[1]
        cols = data.shape[2]
        X = np.zeros((data.shape[0] - configs.total_length + 1, configs.input_length, rows, cols, 1), dtype=np.float32)
        Y = np.zeros(
            (data.shape[0] - configs.total_length + 1, configs.total_length - configs.input_length, rows, cols, 1),
            dtype=np.float32)

        for i in range(data.shape[0]):
            if i <= (data.shape[0] - configs.total_length):
                X[i, ::, ::, ::, 0] = data[i:i + configs.input_length]
                Y[i, ::, ::, ::, 0] = data[i + configs.input_length:i + configs.total_length]
        xy = np.concatenate((X, Y), axis=1)
        xy = list(xy)
        return xy

    def get_batch(self, train_seqs, test_seqs, batch_size, trainShuffle=False):
        if trainShuffle:
            random.seed(self.seed)
            random.shuffle(train_seqs)
        batchs = int(len(train_seqs) / batch_size)
        current_position = 0
        train_batchs = []
        for i in range(batchs):
            batch_data = np.array(train_seqs[current_position:current_position + batch_size])
            train_batchs.append(batch_data)
            current_position += batch_size

        batchs = int(len(test_seqs) / batch_size)
        current_position = 0
        test_batchs = []
        for i in range(batchs):
            batch_data = np.array(test_seqs[current_position:current_position + batch_size])
            test_batchs.append(batch_data)
            current_position += batch_size

        return train_batchs, test_batchs

    def get_sets(self, data_path):
        #alldata shape: [sequence length(9552), width(128), height(128)]
        alldata = list(np.load(data_path))  # load all spatial-maps
        print('dataset loaded:', data_path)
        size = len(alldata)
        train_size = int(size * self.train_p)
        train_data = alldata[:train_size]
        test_data = alldata[train_size:]
        train_seqs = self.split_list(train_data)
        test_seqs = self.split_list(test_data)
        train_set, test_set = self.get_batch(train_seqs, test_seqs, configs.batch_size, trainShuffle=True)
        train_set = np.array(train_set)
        test_set = np.array(test_set)
        train_x, train_y = train_set[:, :, :-1, :, :], train_set[:, :, -1:, :, :]
        test_x, test_y = test_set[:, :, :-1, :, :], test_set[:, :, -1:, :, :]
        train_set = np.concatenate((train_x, train_y), axis=2)
        test_set = np.concatenate((test_x, test_y), axis=2)
        return train_set, test_set

