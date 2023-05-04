import datetime
import time
import utils.validate as validate
from utils.dataloader import Dataset
from utils.earlystop import EarlyStopping
import models.trainer as trainer
import matplotlib as mpl

mpl.use('Agg')
from tqdm import tqdm
from config import configs
from models.model_factory import Model
from utils import preprocess
from utils.helper import adjust_learning_rate, AverageMeter

# -----------------------------------------------------------------------------

args = configs
print('device:', args.device)
print('dataset:', args.data_path)
print('batch_size:', args.batch_size)
print('model_name:', args.model_name)


def train(model):
    early_stopping = EarlyStopping(patience=configs.early_stop)
    if args.pretrained_model:
        model.load(args.pretrained_model)

    # load data
    dataset = Dataset((args.train_p, 1 - args.train_p))
    train_set, test_set = dataset.get_sets(configs.data_path)
    print('train_batchs:', len(train_set), 'test_batchs:', len(test_set))

    llr = args.lr
    model.optimizer = adjust_learning_rate(model.optimizer, llr)  # 调整学习率
    print('start training...')
    for epoch in range(1, args.max_epoch + 1):
        startTime = time.time()
        losses = AverageMeter()
        if epoch % args.adjust_interval == 0 and epoch > 0:  # 每 adjust_interval步长调整学习率
            llr = llr * args.adjust_rate  # 学习率 *= 0.5
            model.optimizer = adjust_learning_rate(model.optimizer, llr)
        for ind, ims in enumerate(tqdm(train_set, ncols=50)):
            ims = preprocess.reshape_patch(ims, args.patch_size)  # ims(4 , 20 , 140//4 , 140//4 , 4*4*1)
            tr_loss = trainer.train(model, ims, args, epoch)
            losses.update(tr_loss)
        val_losses = validate.run(model, test_set)
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), round(time.time() - startTime, 2))
        print('epoch:{0} lr: {lr:.5f} '
              'train_loss:{train_loss.avg:.4f} '
              'val_loss:{val_loss.avg:.4f} '.format(
            epoch, lr=model.optimizer.param_groups[-1]['lr'], train_loss=losses, val_loss=val_losses))

        # 早停机制
        if configs.early_stop:
            early_stopping(val_losses.avg, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 保存模型文件
        if epoch % args.snapshot_interval == 0:
            # 模型性能评估
            model.save(args.model_name + '-e' + str(epoch)
                       + '-t_loss' + str(round(float(losses.avg), 2))
                       + '-v_loss' + str(round(float(val_losses.avg), 2)))

model = Model(args)
p_num = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
print('Total model_params:', p_num)
train(model)
