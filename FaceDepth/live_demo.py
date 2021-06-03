import numpy as np
import cv2
img_size1 = 640
img_size2 = 480
import torch
from FaceDepth_model import Model
cap = cv2.VideoCapture(0)
import torchvision.transforms as transforms
from PIL import Image
model = Model().cuda()
model.load_state_dict(torch.load('weights_model.pth'))
model.eval()

def predict(model, img_array):
    preds = []
    for img in img_array:
        t1 = img.transpose(-1, 0, 1).reshape(1, 3, 480, 640)
        t2 = torch.from_numpy(t1).float()
        t3 = model(t2.cuda())
        t4 = t3.detach().cpu().numpy()
        t5 = t4/25
        preds.append(t5[0][0])
    return preds
import cmapy
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.applyColorMap(gray, cmapy.cmap('viridis'))
    # gray = cv2.applyColorMap(gray, cmapy.cmap('inferno'))
    img_array = cv2.resize(gray, (img_size1, img_size2))
    numpy_array = np.asarray([img_array])
    numpy_array = numpy_array / 255.
    prediction = predict(model, numpy_array)
    prediction = prediction
    prediction = np.squeeze(prediction)
    prediction = prediction.astype(np.uint8)
    prediction = cv2.resize(prediction, (640, 480))
    print(prediction)

    # Display the resulting frame
    cv2.imshow('frame', prediction)
    count = 0
    cv2.imwrite("frame%d.png" % count, prediction)
    count = count + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()