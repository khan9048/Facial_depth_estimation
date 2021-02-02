import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
import cv2
import matplotlib
import matplotlib.cm


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}


def getTrainTestdata(dataPath, trainCount):
    traincsvFilepath = 'data_exr_base_01/wholeFaceDepth/FaceDepthSynth/data_train.csv'  # 'data_exr/data_train.csv'
    testcsvFilepath = 'data_exr_base_01/wholeFaceDepth/FaceDepthSynth/data_test.csv'

    with open(traincsvFilepath, 'r') as f:
        x = [line.rstrip() for line in f]

    my_train_list = list((row.split(',') for row in x if len(row) > 0))

    with open(testcsvFilepath, 'r') as f:
        y = [line.rstrip() for line in f]

    my_test_list = list((row.split(',') for row in y if len(row) > 0))

    # testing
    if True:
        testCount = int(trainCount/5)
        my_train_list = my_train_list[0:trainCount]
        my_test_list = my_test_list[0:testCount]

    return my_train_list, my_test_list


class DepthDataSetMemory(Dataset):
    def __init__(self, nyu2_train, transform=None):
        # self.data = data
        self.trainList = nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.trainList[idx]

        image = cv2.imread(sample[0])
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()

        # read the depth data
        depth = cv2.imread(sample[1], cv2.IMREAD_UNCHANGED)[:, :, 0]
        # import matplotlib.pyplot as plt
        # plt.imshow(depth)
        # plt.show()
        # Reshape in dims
        image = np.asarray(image).reshape(480, 640, 3)
        depth = np.asarray(depth).reshape(480, 640, 1)
        depth = cv2.resize(depth, (320, 240))

        # image = Image.open(BytesIO(self.data[sample[0]]))
        # depth = Image.open(BytesIO(self.data[sample[1]]))
        sample = {'image': image, 'depth': depth}
        if self.transform: sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.trainList)


class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = self.to_tensor(image)

        # depth = cv2.resize(depth, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)  # depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            # depth = self.to_tensor(depth).float() * 1000
            depth = self.to_tensor(depth).float()

        # put in expected range
        depth = torch.clamp(depth, 0, 5)  # sbasak01 - 10, 1000

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            # img = torch.from_numpy(pic.transpose((2, 0, 1)))
            if pic.shape[2] == 3:
                img = torch.from_numpy(pic.transpose(-1, 0, 1))
                return img.float().div(255)
            else:
                img = torch.from_numpy(pic.transpose(-1, 0, 1))
                return img.float().div(5)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])


def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])


def getTrainingTestingData(batch_size, trainCount):
    # data, nyu2_train = loadZipToMem('nyu_data.zip')

    train_list, test_list = getTrainTestdata('testPath', trainCount)

    transformed_training = DepthDataSetMemory(train_list,
                                              transform=getNoTransform())  # transform=getDefaultTrainTransform()
    transformed_testing = DepthDataSetMemory(test_list, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size,
                                                                                  shuffle=False)


def DepthNorm(depth, maxDepth=5.0):  # sbasak01 - 1000
    return maxDepth / depth


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0, :, :]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))
