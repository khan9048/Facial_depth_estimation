import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from Mobile_model import Model


def show_images(images, save=False):
    fig, axis = plt.subplots(3, 4)
    nir = axis[0, 0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    fig.colorbar(nir, ax=axis[0, 0], orientation='horizontal')
    nir1 = axis[0, 1].imshow(cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB))
    fig.colorbar(nir1, ax=axis[0, 1], orientation='horizontal')
    nir2 = axis[0, 2].imshow(cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB))
    fig.colorbar(nir2, ax=axis[0, 2], orientation='horizontal')
    nir3 = axis[0, 3].imshow(cv2.cvtColor(images[3], cv2.COLOR_BGR2RGB))
    fig.colorbar(nir3, ax=axis[0, 3], orientation='horizontal')

    nir4 = axis[1, 0].imshow(cv2.cvtColor(images[4], cv2.COLOR_BGR2RGB))
    fig.colorbar(nir4, ax=axis[1, 0], orientation='horizontal')
    nir5 = axis[1, 1].imshow(cv2.cvtColor(images[5], cv2.COLOR_BGR2RGB))
    fig.colorbar(nir5, ax=axis[1, 1], orientation='horizontal')
    nir6 = axis[1, 2].imshow(cv2.cvtColor(images[6], cv2.COLOR_BGR2RGB))
    fig.colorbar(nir6, ax=axis[1, 2], orientation='horizontal')
    nir7 = axis[1, 3].imshow(cv2.cvtColor(images[7], cv2.COLOR_BGR2RGB))
    fig.colorbar(nir7, ax=axis[1, 3], orientation='horizontal')

    nir8 = axis[2, 0].imshow(cv2.cvtColor(images[8], cv2.COLOR_BGR2RGB))
    fig.colorbar(nir8, ax=axis[2, 0], orientation='horizontal')
    nir9 = axis[2, 1].imshow(cv2.cvtColor(images[9], cv2.COLOR_BGR2RGB))
    fig.colorbar(nir9, ax=axis[2, 1], orientation='horizontal')
    nir10 = axis[2, 2].imshow(cv2.cvtColor(images[10], cv2.COLOR_BGR2RGB))
    fig.colorbar(nir10, ax=axis[2, 2], orientation='horizontal')
    # nir11 = axis[2, 3].imshow(cv2.cvtColor(images[11], cv2.COLOR_BGR2RGB))
    # fig.colorbar(nir11, ax=axis[2, 3])


    plt.show()
    if save:
        plt.savefig('result.png')


def show_depth_colormap(images, save=False):
    fig, axis = plt.subplots(3, 4)
    nir = axis[0, 0].imshow(images[0].squeeze(), cmap='Greys')
    fig.colorbar(nir, ax=axis[0, 0], orientation='horizontal')
    nir1 = axis[0, 1].imshow(images[1].squeeze(), cmap='Greys')
    fig.colorbar(nir1, ax=axis[0, 1], orientation='horizontal')
    nir2 = axis[0, 2].imshow(images[2].squeeze(), cmap='Greys')
    fig.colorbar(nir2, ax=axis[0, 2], orientation='horizontal')
    nir3 = axis[0, 3].imshow(images[3].squeeze(), cmap='Greys')
    fig.colorbar(nir3, ax=axis[0, 3], orientation='horizontal')

    nir4 = axis[1, 0].imshow(images[4].squeeze(), cmap='Greys')
    fig.colorbar(nir4, ax=axis[1, 0], orientation='horizontal')
    nir5 = axis[1, 1].imshow(images[5].squeeze(), cmap='Greys')
    fig.colorbar(nir5, ax=axis[1, 1], orientation='horizontal')
    nir6 = axis[1, 2].imshow(images[6].squeeze(), cmap='Greys')
    fig.colorbar(nir6, ax=axis[1, 2], orientation='horizontal')
    nir7 = axis[1, 3].imshow(images[6].squeeze(), cmap='Greys')
    fig.colorbar(nir7, ax=axis[1, 3], orientation='horizontal')

    nir8 = axis[2, 0].imshow(images[8].squeeze(), cmap='Greys')
    fig.colorbar(nir8, ax=axis[2, 0], orientation='horizontal')
    nir9 = axis[2, 1].imshow(images[9].squeeze(), cmap='Greys')
    fig.colorbar(nir9, ax=axis[2, 1], orientation='horizontal')
    nir10 = axis[2, 2].imshow(images[10].squeeze(), cmap='Greys')
    fig.colorbar(nir10, ax=axis[2, 2], orientation='horizontal')
    # nir11 = axis[2, 3].imshow(images[11].squeeze(), cmap='Greys')
    # fig.colorbar(nir11, ax=axis[2, 3])

    plt.show()
    if save:
        plt.savefig('result.png')

def load_images(image_list):
    loaded_images = []
    for file in image_list:
        # x = np.clip(np.array(cv2.imread(file), dtype=float) / 255, 0, 1)
        x = np.asarray(cv2.imread(file))
        # x = cv2.resize(x, (640, 480))
        loaded_images.append(x)

    return loaded_images  # np.stack(loaded_images, axis=0)


def predict(model, test_images):
    preds = []
    for img in test_images:
        t1 = img.transpose(-1, 0, 1).reshape(1, 3, 480, 640)
        t2 = torch.from_numpy(t1).float().div(255)
        t3 = model(t2.cuda())
        t4 = t3.detach().cpu().numpy()
        t5 = t4
        # t5 = cv2.resize(t5, (320, 240))
        preds.append(t5[0][0])
    return preds


image_list = glob.glob('rgb_syn_test/*.jpg')
test_images = load_images(image_list)

image_list_gt = glob.glob('gt_syn_test/*.png')
test_images_gt = load_images(image_list_gt)
# model checkpoints
model = Model().cuda()
# model.load_state_dict(torch.load(r'models\01-27-2021_12-51-33-n17183-e20-bs4-lr0.0001\weights.epoch8_model.pth'))
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

optimizer = torch.optim.Adam(model.parameters(), 0.0001)
checkpoint = torch.load(r'models\01-27-2021_12-51-33-n17183-e20-bs4-lr0.0001\weights.epoch8_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()

predictions = predict(model, test_images)

show_images(test_images)
show_depth_colormap(test_images_gt)
show_depth_colormap(predictions)

i = 1
# save the prediction in numpy file
for pred in predictions:
    plt.imsave("gt_syn_test/depth_image_{0}.jpg".format(i),pred,cmap= 'Greys')
    i += 1