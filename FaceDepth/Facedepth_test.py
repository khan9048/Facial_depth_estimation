import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from FaceDepth_model import Model


def show_images(images, save=False):
    fig, axis = plt.subplots(3, 4)
    axis[0, 0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    axis[0, 1].imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
    axis[0, 2].imshow(cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB))
    axis[0, 3].imshow(cv2.cvtColor(images[3], cv2.COLOR_BGR2RGB))

    axis[1, 0].imshow(cv2.cvtColor(images[4], cv2.COLOR_BGR2RGB))
    axis[1, 1].imshow(cv2.cvtColor(images[5], cv2.COLOR_BGR2RGB))
    axis[1, 2].imshow(cv2.cvtColor(images[6], cv2.COLOR_BGR2RGB))
    axis[1, 3].imshow(cv2.cvtColor(images[7], cv2.COLOR_BGR2RGB))

    axis[2, 0].imshow(cv2.cvtColor(images[8], cv2.COLOR_BGR2RGB))
    axis[2, 1].imshow(cv2.cvtColor(images[9], cv2.COLOR_BGR2RGB))
    # axis[2, 2].imshow(cv2.cvtColor(images[10], cv2.COLOR_BGR2RGB))
    # axis[2, 3].imshow(cv2.cvtColor(images[11], cv2.COLOR_BGR2RGB))


    plt.show()
    if save:
        plt.savefig('result.png')


def show_depth_colormap(images, save=False):
    fig, axis = plt.subplots(3, 4)
    axis[0, 0].imshow(images[0].squeeze(), cmap='Greys')
    axis[0, 1].imshow(images[1].squeeze(), cmap='Greys')
    axis[0, 2].imshow(images[2].squeeze(), cmap='Greys')
    axis[0, 3].imshow(images[3].squeeze(), cmap='Greys')

    axis[1, 0].imshow(images[4].squeeze(), cmap='Greys')
    axis[1, 1].imshow(images[5].squeeze(), cmap='Greys')
    axis[1, 2].imshow(images[6].squeeze(), cmap='Greys')
    axis[1, 3].imshow(images[6].squeeze(), cmap='Greys')

    axis[2, 0].imshow(images[8].squeeze(), cmap='Greys')
    axis[2, 1].imshow(images[9].squeeze(), cmap='Greys')
    # axis[2, 2].imshow(images[10].squeeze(), cmap='Greys')
    # axis[2, 3].imshow(images[11].squeeze(), cmap='Greys')

    plt.show()
    if save:
        plt.savefig('result.png')

def load_images(image_list):
    loaded_images = []
    for file in image_list:
        x = np.asarray(cv2.imread(file))
        loaded_images.append(x)

    return loaded_images


def predict(model, test_images):
    preds = []
    for img in test_images:
        t1 = img.transpose(-1, 0, 1).reshape(1, 3, 480, 640)
        t2 = torch.from_numpy(t1).float().div(255)
        t3 = model(t2.cuda())
        t4 = t3.detach().cpu().numpy()
        t5 = t4/150
        preds.append(t5[0][0])
    return preds


image_list = glob.glob('rgb_syn_test\*.jpg')
test_images = load_images(image_list)

image_list_gt = glob.glob('gt_syn_test\*.png')
test_images_gt = load_images(image_list_gt)
# Model checkpiont loding

model = Model().cuda()
model.load_state_dict(torch.load('weights_model.pth'))
model.eval()

predictions = predict(model, test_images)

show_images(test_images)
show_depth_colormap(test_images_gt)
show_depth_colormap(predictions)

i = 1
# save the prediction in numpy file
for pred in predictions:
    plt.imsave("pre_syn_test\depth_image_{0}.jpg".format(i),pred,cmap= 'Greys')
    i += 1