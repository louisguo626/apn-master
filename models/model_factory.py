import os
import numpy as np
import torch
from torch.optim import Adam
from models import predrnn_v2, mim, convlstm, air_prednet


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn_v2': predrnn_v2.RNN,
            'mim': mim.MIM,
            'convlstm': convlstm.RNN,
            'air_prednet': air_prednet.RNN
        }

        self.mask = np.zeros((configs.batch_size,  # zeros->(4 , 20-10-1 , 140//4 , 140//4 , 4^2)
                              configs.total_length - configs.input_length - 1,
                              configs.img_width // configs.patch_size,
                              configs.img_width // configs.patch_size,
                              configs.patch_size ** 2 * configs.img_channel))

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)

    def save(self, fileName):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, fileName + '.pth')
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(self.mask).to(self.configs.device)
        self.optimizer.zero_grad()
        next_frames, loss = self.network(frames_tensor, mask_tensor)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(self.mask).to(self.configs.device)
        next_frames, _ = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()
