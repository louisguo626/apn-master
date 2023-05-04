import argparse

parser = argparse.ArgumentParser(description='PyTorch AirPredNet')   #添加系统参数
# training/test
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_path', type=str, default='data/idw/alldata_idw_new.npy')
parser.add_argument('--save_dir', type=str, default='checkpoints/')
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--train_p', type=float, default=0.9)

# input & output size
parser.add_argument('--input_length', type=int, default=6)
parser.add_argument('--total_length', type=int, default=12)
parser.add_argument('--img_width', type=int, default=128)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='mim')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64, 64, 64')  # 64, 64, 64, 64
parser.add_argument('--filter_size', type=int, default=[5, 5, 5])
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--decouple_beta', type=float, default=0.1)

# optimization
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--early_stop', type=int, default=7)
parser.add_argument('--adjust_interval', type=int, default=75)#用来调整学习率的
parser.add_argument('--adjust_rate', type=float, default=0.5)
parser.add_argument('--snapshot_interval', type=int, default=1)
parser.add_argument('--layer_norm', type=int, default=0)

configs = parser.parse_args()
