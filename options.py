import argparse
from utils import str2bool

parser = argparse.ArgumentParser()

# device settings
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

# dataset settings
parser.add_argument('--image_root', type=str, default='/home/omnisky/diskB/datasets/SOD_Datasets/DUTS-TR',
                    help='the training images root')
parser.add_argument('--video_root', type=str, default='/home/omnisky/diskB/datasets/sal_video_datasets/VSOD-TR',
                    help='the training videos root')
parser.add_argument('--test_image_root', type=str, default='/home/omnisky/diskB/datasets/SOD_Datasets/DUTS-TE',
                    help='the test images root')
parser.add_argument('--test_video_root', type=str, default='/home/omnisky/diskB/datasets/sal_video_datasets/DAVSOD-V',
                    help='the test videos root')
parser.add_argument('--trainsize', type=int, default=352, help='pretrain image size')
parser.add_argument('--num_workers', type=int, default=4)

# pretrain settings
parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay
parser.add_argument('--clip', type=float, default=-1, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')

# finetune settings

parser.add_argument('--epoch', type=int, default=15, help='pretrain batch size')
parser.add_argument('--lr', type=float, default=2e-6, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='pretrain batch size')
parser.add_argument('--decay_epoch', type=int, default=10, help='every n epochs decay learning rate')

# architecture settings
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--norm', type=str, default='gn', choices=['bn', 'gn'])
parser.add_argument('--loss_type', type=str, default='bas', choices=['bce', 'bi', 'bas', 'f3'])

# save settings
parser.add_argument('--save_path', type=str, default='/home/omnisky/diskB/SCANet_work_dir',
                    help='the path to save models and logs')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')

# test settings
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--test_path', type=str, default=r'E:\sal_video_datasets', help='test dataset path')
parser.add_argument('--save_root', type=str, default='./SCANet_work_dir/run-0', help='test dataset saving path')
parser.add_argument('--pretrained', type=str, default='', help='load pretrained weight')
opt = parser.parse_args()
