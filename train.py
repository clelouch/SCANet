import time
import datetime
import os
import torch
import numpy as np
from datetime import datetime, timedelta
from torchvision.utils import make_grid
from model import FinetuneNet
from dataset import get_loader, TestDataset
from utils import clip_gradient, setup_seed
from loss import build_loss
import logging
import torch.backends.cudnn as cudnn

from options import opt
from logger import Logger

experiment_name = "SCANet"
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
writer = Logger("{}/logs/{}".format(opt.save_path, experiment_name),
                clear=True, port=8000, palette=palette)

with open('%s/args.txt' % (opt.save_path), 'w') as f:
    for arg in vars(opt):
        print('%s: %s' % (arg, getattr(opt, arg)), file=f)

# set the device for training
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU ' + opt.gpu_id)
cudnn.benchmark = True

# build the model
model = FinetuneNet(opt).cuda()
model.encoder.load_from(np.load('./models/imagenet21k_R50+ViT-B_16.npz'))

if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,
                             weight_decay=opt.wd)

# set the path
img_root = opt.image_root
video_root = opt.video_root
test_video_root = opt.test_video_root
save_path = opt.save_path

if os.path.exists(os.path.join(save_path, 'models')):
    raise Exception("directory exists! Please change save path")
if not os.path.exists(os.path.join(save_path, 'models')):
    os.makedirs(os.path.join(save_path, 'models'))

# load data
print('load data...')
train_loader = get_loader(img_root, video_root, batch_size=opt.batchsize, train_size=opt.trainsize,
                          num_workers=opt.num_workers)
test_loader = TestDataset(test_video_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=os.path.join(save_path, 'log.log'),
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("SCANet-train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

step = 0
best_mae = 1
best_epoch = 0
log_interval = 500
criterior = build_loss(opt.loss_type)


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        start_time = time.time()
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            gts = gts.view(gts.size(0) * gts.size(1), gts.size(2), gts.size(3), gts.size(4))

            pred1, pred2, pred3, pred4 = model(images)
            loss1 = criterior(pred1, gts)
            loss2 = criterior(pred2, gts)
            loss3 = criterior(pred3, gts)
            loss4 = criterior(pred4, gts)

            loss = loss1 + loss2 / 2 + loss3 / 4 + loss4 / 8
            loss.backward()

            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % log_interval == 0:
                end_time = time.time()
                duration_time = end_time - start_time
                time_second_avg = duration_time / (opt.batchsize * log_interval)
                eta_sec = time_second_avg * (
                        (opt.epoch - epoch - 1) * len(train_loader) * opt.batchsize + (
                        len(train_loader) - i - 1) * opt.batchsize
                )
                eta_str = str(timedelta(seconds=int(eta_sec)))
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], eta: {}, AllLoss: {:.5f} Loss1: {:0.5f}'.
                      format(datetime.now(), epoch + 1, opt.epoch, i, total_step, eta_str, loss.data, loss1.data))
                logging.info(
                    '#Train # :Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], eta: {}, AllLoss: {:.5f} Loss1: {:0.5f}'.
                        format(epoch + 1, opt.epoch, i, total_step, eta_str, loss.data, loss1.data))
                writer.add_scalar('Loss', loss.cpu().data, step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = pred1[0].clone()
                res = res.data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_label('s1', torch.tensor(res), step)
                start_time = time.time()

        loss_all /= epoch_step
        logging.info('#Train #: Epoch [{:03d}/{:03d}], Loss_AVG: {:.5f}'.format(epoch + 1, opt.epoch,
                                                                                   loss_all))
        writer.add_scalar('Loss-epoch', loss_all.cpu(), step)
        if (epoch + 1) % 3 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'models', 'epoch_{}.pth'.format(epoch + 1)))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'models', 'epoch_{}.pth'.format(epoch + 1)))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res = model(image)[0]
            # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            for index in range(5):
                res[index] = (res[index] - res[index].min()) / (res[index].max() - res[index].min() + 1e-8)
                mae_sum += np.sum(np.abs(res[index] - gt[index])) * 1.0 / (gt.shape[1] * gt.shape[2])
        mae = mae_sum / (test_loader.size * 5)
        writer.add_scalar('MAE', torch.tensor(mae), epoch)

        if epoch == 0:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, 'models', 'best.pth'))
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch + 1, mae, best_epoch + 1, best_mae))
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch + 1, mae, best_mae, best_epoch + 1))


if __name__ == '__main__':
    print("Start train...")
    setup_seed()
    # decay_epochs = [10, 15]
    for epoch in range(opt.epoch):
        writer.add_scalar('learning_rate', opt.lr, epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)

        if (epoch + 1) == 10:
            opt.lr = opt.lr * 0.1
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,
                                             weight_decay=opt.wd)
