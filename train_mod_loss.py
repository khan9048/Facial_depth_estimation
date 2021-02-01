import time
import argparse
import datetime
import pathlib

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils.functions import imgrad_yx
from unet import UNet
from loss import ssim, RMSE, RMSE_log, GradLoss, NormalLoss
from utils.data import getTrainingTestingData, AverageMeter, DepthNorm, colorize


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    parser.add_argument('--o', dest='optimizer', help='training optimizer', default="adam", type=str)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', help='step to do learning rate decay, unit is epoch', default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', help='learning rate decay ratio', default=0.1, type=float)
    args = parser.parse_args()

    # Create model
    # model = Model().cuda()
    model = UNet(n_channels=3, n_classes=1, bilinear=True).cuda()
    # model.load_state_dict(torch.load(r'models\06-05-2020_15-04-20-n15000-e10-bs4-lr0.0001\weights.epoch9_model.pth'))
    print('Model created.')

    print(summary(model, (3, 480, 640)))

    # hyperparams
    lr = args.lr
    bs = args.bs
    lr_decay_step = args.lr_decay_step
    lr_decay_gamma = args.lr_decay_gamma
    DOUBLE_BIAS = True
    WEIGHT_DECAY = 0.0001

    # params
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (DOUBLE_BIAS + 1), \
                            'weight_decay': 4e-5 and WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': 4e-5}]

    # optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size, trainCount=60)

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Loss
    # l1_criterion = nn.L1Loss()
    # rmse = RMSE()
    l1_criterion = nn.L1Loss()
    grad_criterion = GradLoss()
    normal_criterion = NormalLoss()
    # eval_metric = RMSE_log()

    now = datetime.datetime.now()  # current date and time
    runID = now.strftime("%m-%d-%Y_%H-%M-%S") + '-n' + str(len(train_loader)) + '-e' + str(args.epochs) + '-bs' + str(
        batch_size) + '-lr' + str(args.lr)
    outputPath = './models/'
    runPath = outputPath + runID
    pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)

    # constants
    # grad_factor = 10.
    # normal_factor = 10.

    # Start training...
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # Normalize depth
            depth_n = DepthNorm(depth)

            # Predict
            output = model(image)

            # Compute the loss
            l_depth = l1_criterion(output, depth_n)
            # l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range=5.0)) * 0.5, 0, 1)  # sbasak01

            # sbasak01 loss modification
            grad_real, grad_fake = imgrad_yx(depth_n), imgrad_yx(output)
            grad_loss = grad_criterion(grad_fake, grad_real) #* grad_factor  # * (epoch > 3)
            normal_loss = normal_criterion(grad_fake, grad_real) #* normal_factor  # * (epoch > 7)

            # loss = (1.0 * l_ssim) + (0.1 * l_depth)

            loss = (l_ssim) + (l_depth) + (grad_loss) + (normal_loss)

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))

            # Log progress
            niter = epoch * N + i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'L1_Loss: {l1_loss:.4f} SSIM_Loss: {ssim_loss:.4f} grad_loss: {gradloss:.4f} normal_loss: {'
                      'normalloss:.4f} '
                      .format(epoch, i, N,
                              batch_time=batch_time,
                              loss=losses,
                              eta=eta,
                              l1_loss=l_depth,
                              ssim_loss=l_ssim,
                              gradloss=grad_loss,  # sbasak01 loss modification
                              normalloss=normal_loss  # sbasak01 loss modification
                              ))

        #         # Log to tensorboard
        #         writer.add_scalar('Train/Loss', losses.val, niter)
        #
        # writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

        # save Model intermediate
        path = runPath + '/weights.epoch{0}_model.pth'.format(epoch)
        # torch.save(model.cpu().state_dict(), path)  # saving model
        # model.cuda()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses,
        }, path)
        model.cuda()
import torchvision.utils as vutils
def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)),
                                    epoch)
    output = DepthNorm(model(image))
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff',
                     colorize(vutils.make_grid(torch.abs(output - depth).data, nrow=6, normalize=False)), epoch)
    del image
    del depth
    del output


if __name__ == '__main__':
    main()
