import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from model import FinetuneNet
from dataset import InferDataset
import time
from torch.backends import cudnn
from options import opt

dataset_path = opt.test_path

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU ' + opt.gpu_id)
cudnn.benchmark = True

# load the model
model = FinetuneNet(config=opt)
model.load_state_dict(torch.load(opt.pretrained))
model.cuda()
model.eval()

test_datasets = ['DAVIS-TE', 'DAVSOD-TE', 'FBMS', 'SegTrack-V2', 'ViSal', 'VOS-TE']
with torch.no_grad():
    for dataset in test_datasets:
        save_path = os.path.join(opt.save_root, dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        video_root = os.path.join(dataset_path, dataset)
        test_loader = InferDataset(video_root, opt.testsize)
        for i in range(test_loader.size):
            image, sizes, name1, name2 = test_loader.load_data()
            image = image.cuda()
            res = model(image)[0]
            res = F.interpolate(res, size=(sizes[0][1], sizes[0][0]), mode='bilinear',
                                     align_corners=True)
            for index in range(0, 5):
                # print('save img to: ', name1[hh])
                save_img = res[index]
                save_img = (save_img - save_img.min()) / (save_img.max() - save_img.min() + 1e-8)
                save_img = save_img.data.cpu().numpy().squeeze()
                save_path_for_each_image = os.path.join(save_path, name1[index], name2[index])
                if not os.path.exists(os.path.join(save_path, name1[index])):
                    os.makedirs(os.path.join(save_path, name1[index]))
                cv2.imwrite(save_path_for_each_image, save_img * 255)
        print(dataset, ' Test Done!')
