import torch
import torch.nn as nn
import torchvision
import time
import argparse
import datetime
import sys
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from utils.functions import imgrad_yx
import pathlib
from torchvision import transforms
from logger import Logger
from torchsummary import summary
from utils.data import getTrainingTestingData, AverageMeter, DepthNorm, colorize
from loss import ssim, RMSE, RMSE_log, GradLoss, NormalLoss
from unet import UNet
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
    writer_train = SummaryWriter('runs/train_0')
    # model.load_state_dict(torch.load(r'models\06-05-2020_15-04-20-n15000-e10-bs4-lr0.0001\weights.epoch9_model.pth'))
    print('Model created.')
    logger = Logger('./logs')

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
    prefix = 'unet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size, trainCount=68730)

    # # Logging
    # writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

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
            grad_loss = grad_criterion(grad_fake, grad_real)  # * grad_factor  # * (epoch > 3)
            normal_loss = normal_criterion(grad_fake, grad_real)  # * normal_factor  # * (epoch > 7)

            # loss = (1.0 * l_ssim) + (0.1 * l_depth)

            loss = (l_ssim) + (l_depth) + (grad_loss) + (normal_loss)

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()
            # ADD TRAINING LOSS TO SummaryWriter
            writer_train.add_scalar('LOSS', loss.data.item(), i)

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

                # # 1. Log scalar values (scalar summary)
                # info = {'loss': loss.item()}
                #
                # for tag, value in info.items():
                #     logger.scalar_summary(tag, value, i + 1)
                #
                # # 2. Log values and gradients of the parameters (histogram summary)
                # for tag, value in model.named_parameters():
                #     tag = tag.replace('.', '/')
                #     logger.histo_summary(tag, value.data.cpu().numpy(), i + 1)
                #     logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), i + 1)

            if i % 100 == 0:
                sys.stdout.write("Iterations: %d   \r" % (i))
                sys.stdout.flush()
        # save Model intermediate
        path = runPath + '/weights.epoch{0}_model.pth'.format(epoch)
        torch.save(model.cpu().state_dict(), path)  # saving model
        model.cuda()

if __name__ == '__main__':
    main()
