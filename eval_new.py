import numpy as np
import time

import torch
from torchsummary import summary

from Mobile_model import Model

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    return a1, a2, a3, abs_rel, sq_rel, rmse, rmse_log, log_10

def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)

def predict(model, images, batch_size=2, MaxDepth=5):
    MaxDepth = MaxDepth * MaxDepth
    x1 = images.transpose(0, -1, 1, 2)
    x2 = torch.from_numpy(x1).float().div(255)
    x3 = model(x2.cuda())
    x4 = x3.permute(0, 2, 3, 1).detach().cpu().numpy()
    outputs = MaxDepth / x4
    return outputs


def evaluate(model, rgb, depth, crop, batch_size=7, verbose=False):
    N = len(rgb)

    bs = batch_size

    predictions = []
    testSetDepths = []

    for i in range(N // bs):
        x = rgb[(i) * bs:(i + 1) * bs, :, :, :]

        # Compute results
        true_y = depth[(i) * bs:(i + 1) * bs, :, :]
        pred_y = scale_up(2, predict(model, x/255, MaxDepth=5, batch_size=bs)[:, :, :, 0]) * 5.0

        # sbasak01 start masking

        _mask = np.asarray(true_y < 0.210, dtype=np.uint8)

        true_y_Masked = true_y * _mask
        true_y_Masked[true_y_Masked == 0] = 10
        pred_y_Masked = pred_y * _mask
        pred_y_Masked[pred_y_Masked == 0] = 10

        true_y = true_y_Masked
        pred_y = pred_y_Masked

        # sbasak01 end masking

        # Test time augmentation: mirror image estimate
        # pred_y_flip = scale_up(1,
        #                        predict(model, x[..., ::-1, :] / 255, MaxDepth=5, batch_size=bs)[:, :, :,
        #                        0])  # * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
        pred_y = pred_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
        # pred_y_flip = pred_y_flip[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

        # Compute errors per image in batch
        for j in range(len(true_y)):
            # predictions.append((0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j])))
            predictions.append(pred_y[j])
            testSetDepths.append(true_y[j])

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    e = compute_errors(predictions, testSetDepths)

    return e

rgb = np.load('eigen_test_rgb.npy')
depth = np.load('eigen_test_depth.npy')
crop = np.load('eigen_test_crop.npy')

print('Test data loaded.\n')
from Mobile_model import Model
# model = UNet(n_channels=3, n_classes=1, bilinear=True).cuda()
model = Model().cuda()
# model.load_state_dict(torch.load(r'models_fk_comp_sum_weight=0.28,0.22,0.30,0.20\12-03-2020_06-42-02-n17183-e20-bs4-lr0.0001\weights.epoch7_model.pth'))
# model.eval()
optimizer = torch.optim.Adam(model.parameters(), 0.0001)
checkpoint = torch.load(r'models\01-27-2021_12-51-33-n17183-e20-bs4-lr0.0001\weights.epoch8_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()
print(summary(model, (3, 240, 320)))

start = time.time()
print('Testing...')

e = evaluate(model, rgb, depth, crop, batch_size=7)
print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log_10'))
print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]))


# print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'log10'))
# print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2],
#         e[3], e[4], e[5], e[6], e[7]))


end = time.time()
print('\nTest time', end - start, 's')