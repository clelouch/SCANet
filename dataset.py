import os
import torch
from torch.utils import data
import random
import numpy as np
from torchvision import transforms
from PIL import Image
from PIL import ImageEnhance


def random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def random_crop(img, label):
    border = 30
    img_width = img.size[0]
    img_height = img.size[1]
    crop_width = np.random.randint(img_width - border, img_width)
    crop_height = np.random.randint(img_height - border, img_height)
    left_bound = (img_width - crop_width) // 2
    up_bound = (img_height - crop_height) // 2

    random_region = (left_bound, up_bound, left_bound + crop_width, up_bound + crop_height)
    return img.crop(random_region), label.crop(random_region)


def random_rotate(img, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        img = img.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return img, label


def color_enhance(img):
    bright_intensity = random.randint(5, 15) / 10.0
    img = ImageEnhance.Brightness(img).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    img = ImageEnhance.Contrast(img).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    img = ImageEnhance.Color(img).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    img = ImageEnhance.Sharpness(img).enhance(sharp_intensity)
    return img


def random_gaussian(img, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(img)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def random_peper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


def rgb_loader(img_path):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def gt_loader(gt_path):
    with open(gt_path, 'rb') as f:
        gt = Image.open(f)
        return gt.convert('L')


def resize_img_gt(img, gt):
    assert img.size == gt.size, str(img.size) + " != " + str(gt.size) + ", sizes not equal"


def filter_files(imgs, labels):
    assert len(imgs) == len(labels), str(len(imgs)) + ' != ' + str(len(labels)) + "length not equal"
    images = []
    gts = []
    for img_path, label_path in zip(imgs, labels):
        img = Image.open(img_path)
        label = Image.open(label_path)
        if img.size == label.size:
            images.append(img_path)
            gts.append(label_path)
    return images, gts


def make_image_dataset(img_root):
    image_names = os.listdir(os.path.join(img_root, 'images'))
    gt_names = os.listdir(os.path.join(img_root, 'gts'))
    image_names = sorted(image_names)
    gt_names = sorted(gt_names)

    image_paths = []
    gt_paths = []
    for i in range(len(image_names)):
        image_paths.append(os.path.join(img_root, 'images', image_names[i]))
        gt_file = os.path.join(img_root, 'gts', image_names[i][:-4] + '.jpg')
        if not os.path.exists(gt_file):
            gt_file = os.path.join(img_root, 'gts', image_names[i][:-4] + '.png')
        gt_paths.append(gt_file)
    return image_paths, gt_paths


def make_video2image_dataset(video_root):
    img_paths = []
    gt_paths = []
    cls = os.listdir(video_root)

    for i in range(len(cls)):
        video_dir = os.path.join(video_root, cls[i])
        img_root = os.path.join(video_root, cls[i], 'Imgs')
        gt_root = os.path.join(video_root, cls[i], 'ground-truth')
        if not os.path.exists(gt_root):
            gt_root = os.path.join(video_root, cls[i], 'GT_object_level')
        img_names = os.listdir(img_root)
        gt_names = os.listdir(gt_root)
        img_names = sorted(img_names)
        gt_names = sorted(gt_names)
        for name in img_names:
            img_paths.append(os.path.join(img_root, name))
            gt_paths.append(os.path.join(gt_root, name[:-4] + '.png'))
    return img_paths, gt_paths


def make_video_dataset(video_root):
    all_img_paths = []
    all_gt_paths = []
    all_cls_names = []
    img_paths = []
    gt_paths = []
    cls_names = []
    cls = os.listdir(video_root)

    for i in range(len(cls)):
        videos_dir = os.path.join(video_root, cls[i])
        img_root = os.path.join(videos_dir, 'Imgs')
        gt_root = os.path.join(videos_dir, 'ground-truth')
        if os.path.exists(gt_root) is False:
            gt_root = os.path.join(videos_dir, 'GT_object_level')
        img_names = os.listdir(img_root)
        gt_names = os.listdir(gt_root)

        img_names = sorted(img_names)
        gt_names = sorted(gt_names)
        for j in range(len(img_names)):
            img_paths.append(os.path.join(img_root, img_names[j]))
            gt_paths.append(os.path.join(gt_root, gt_names[j]))
            cls_names.append(cls[i])
        all_img_paths.append(img_paths)
        img_paths = []
        all_gt_paths.append(gt_paths)
        gt_paths = []
        all_cls_names.append(cls_names)
        cls_names = []
    return all_img_paths, all_gt_paths, all_cls_names


class TrainDataset(data.Dataset):
    def __init__(self, img_root, video_root, train_size):
        super(TrainDataset, self).__init__()
        self.train_size = train_size
        self.imgs, self.labels, self.videos = make_video_dataset(video_root)
        self.imgs = sorted(self.imgs)
        self.labels = sorted(self.labels)

        self.static_imgs, self.static_gts = make_image_dataset(img_root)
        self.static_imgs = sorted(self.static_imgs)
        self.static_gts = sorted(self.static_gts)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor()])
        self.video_frame_length = self.compute_video_frame_number()
        self.static_frame_length = self.compute_static_frame_number()
        self.class_index = 0
        self.count = 0

    def compute_video_frame_number(self):
        total_length = 0
        for i in self.imgs:
            total_length += len(i) - 8
        return total_length

    def compute_static_frame_number(self):
        return len(self.static_imgs)

    def get_video_frames(self, index):
        index = self.get_index(index) + 4
        interval = random.randint(1, 2)
        video_features_batch = torch.zeros(5, 3, self.train_size, self.train_size)
        gt_features_batch = torch.zeros(5, 1, self.train_size, self.train_size)
        cls_names = []

        # data augmentation
        random_flip_seed = random.randint(0, 1)
        random_left_bound = np.random.randint(0, 30) // 2
        random_up_bound = np.random.randint(0, 30) // 2
        random_angle = np.random.randint(-15, 15)
        random_rotate_seed = random.random()
        rotate_mode = Image.BICUBIC

        for i in range(-2, 3):
            index_h = index + i * interval
            # print(self.imgs[self.class_index][index_h])
            video_feature = rgb_loader(self.imgs[self.class_index][index_h])
            gt_feature = gt_loader(self.labels[self.class_index][index_h])

            # augmentation
            if random_flip_seed == 1:
                video_feature = video_feature.transpose(Image.FLIP_LEFT_RIGHT)
                gt_feature = gt_feature.transpose(Image.FLIP_LEFT_RIGHT)
            random_region = (random_left_bound, random_up_bound, video_feature.size[0] - random_left_bound,
                             video_feature.size[1] - random_up_bound)
            video_feature, gt_feature = video_feature.crop(random_region), gt_feature.crop(random_region)
            if random_rotate_seed > 0.8:
                video_feature = video_feature.rotate(random_angle, rotate_mode)
                gt_feature = gt_feature.rotate(random_angle, rotate_mode)

            gt_feature = self.gt_transform(gt_feature)
            gt_features_batch[2 + i, :, :, :] = gt_feature
            video_feature = color_enhance(video_feature)
            video_feature = self.img_transform(video_feature)
            video_features_batch[2 + i, :, :, :] = video_feature

            # cls_names.append(self.videos[self.class_index][index_h])
        return video_features_batch, gt_features_batch

    def get_static_frames(self, index):
        img_path = self.static_imgs[index]
        gt_path = self.static_gts[index]
        img = rgb_loader(img_path)
        label = gt_loader(gt_path)
        img, label = random_flip(img, label)
        img, label = random_crop(img, label)
        img, label = random_rotate(img, label)

        img = color_enhance(img)
        img = self.img_transform(img)
        label = self.gt_transform(label)

        img = img.unsqueeze(0).repeat(5, 1, 1, 1)
        label = label.unsqueeze(0).repeat(5, 1, 1, 1)

        return img, label

    def get_index(self, index):
        for i in range(len(self.imgs)):
            if index < len(self.imgs[i]) - 8:
                self.class_index = i
                break
            else:
                index = index - (len(self.imgs[i]) - 8)
        return index

    def __getitem__(self, index):
        if index < self.video_frame_length:
            return self.get_video_frames(index)
        else:
            return self.get_static_frames(index - self.video_frame_length)

    def __len__(self):
        return self.video_frame_length + self.static_frame_length


def get_loader(img_root, video_root, batch_size, train_size, shuffle=True, num_workers=0):
    dataset = TrainDataset(img_root=img_root, video_root=video_root, train_size=train_size)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader


def make_test_dataset(root_path):
    path_listimg = []
    path_listgt = []
    list_cls_names = []
    list_names = []
    frames = 5
    cls = os.listdir(root_path)
    for index in range(0, len(cls)):
        itemimg = []
        itemgt = []
        cls_names = []
        names = []
        image_path = os.path.join(root_path, cls[index], 'Imgs')
        filenames = os.listdir(image_path)
        iters = len(filenames) // (frames)
        iter = 0
        for iter in range(0, iters):
            one_name = filenames[iter * (frames):iter * (frames) + frames]
            i = 1
            for fi in one_name:
                pimg = os.path.join(root_path, cls[index], 'Imgs', fi)
                itemimg.append(pimg)
                pgt = os.path.join(root_path, cls[index], 'GT_object_level', fi[:-4] + '.png')
                if os.path.exists(pgt) is False:
                    pgt = os.path.join(root_path, cls[index], 'ground-truth', fi[:-4] + '.png')
                itemgt.append(pgt)
                cls_names.append(cls[index])
                names.append(fi[:-4] + '.png')
                if i % frames == 0 and i > 0:
                    path_listimg.append(itemimg)
                    path_listgt.append(itemgt)
                    list_cls_names.append(cls_names)
                    list_names.append(names)
                    itemimg = []
                    itemgt = []
                    cls_names = []
                    names = []
                    break
                i = i + 1
        if iter == (iters - 1):
            one_name = filenames[len(filenames) - frames:len(filenames)]
            i = 1
            for fi in one_name:
                if fi.endswith('.jpg') or fi.endswith('.png'):
                    pimg = os.path.join(root_path, cls[index], 'Imgs', fi)
                    itemimg.append(pimg)
                    pgt = os.path.join(root_path, cls[index], 'GT_object_level', fi[:-4] + '.png')
                    if os.path.exists(pgt) is False:
                        pgt = os.path.join(root_path, cls[index], 'ground-truth', fi[:-4] + '.png')
                    itemgt.append(pgt)
                    cls_names.append(cls[index])
                    names.append(fi[:-4] + '.png')
                    if i % frames == 0:
                        path_listimg.append(itemimg)
                        path_listgt.append(itemgt)
                        list_cls_names.append(cls_names)
                        list_names.append(names)
                        break
                    i = i + 1
    return path_listimg, path_listgt, list_cls_names, list_names


class TestDataset:
    def __init__(self, video_root, train_size):
        super(TestDataset, self).__init__()
        self.train_size = train_size
        self.images, self.gts, _, _ = make_test_dataset(video_root)
        # self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        self.image_root = video_root
        self.transform = transforms.Compose([
            transforms.Resize((train_size, train_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((train_size, train_size)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        pathim = self.images[self.index]
        pathgt = self.gts[self.index]
        video_features_batch = torch.zeros(1, 5, 3, self.train_size, self.train_size)
        gt_features_batch = torch.zeros((5, self.train_size, self.train_size))
        for mm in range(0, 5):
            video_features = rgb_loader(pathim[mm])
            video_features = self.transform(video_features).unsqueeze(0)
            video_features_batch[0, mm, :, :, :] = video_features
            gt_features = gt_loader(pathgt[mm])
            gt_features = self.gt_transform(gt_features).squeeze()
            gt_features_batch[mm, :, :] = gt_features

        self.index += 1
        self.index = self.index % self.size
        return video_features_batch, gt_features_batch

    def __len__(self):
        return self.size


class InferDataset:
    def __init__(self, video_root, train_size):
        self.train_size = train_size
        self.images, self.gts, self.cls_names, self.names = make_test_dataset(video_root)
        self.transform = transforms.Compose([
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        pathim = self.images[self.index]
        iter_cls_names = self.cls_names[self.index]
        iter_names = self.names[self.index]
        img_sizes = []
        video_features_batch = torch.zeros(1, 5, 3, self.train_size, self.train_size)
        # print(pathim)
        for mm in range(0, 5):
            video_features = rgb_loader(pathim[mm])
            img_sizes.append(video_features.size)
            video_features = self.transform(video_features)
            video_features_batch[0, mm, :, :, :] = video_features

        self.index += 1
        self.index = self.index % self.size
        return video_features_batch, img_sizes, iter_cls_names, iter_names

    def __len__(self):
        return self.size
