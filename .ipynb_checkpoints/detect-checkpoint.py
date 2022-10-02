import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.image as mat_img
import matplotlib.pyplot as plt
import numpy as np
import gdown

class pneumonia_detector(nn.Module):
    def __init__(self):
        super(pneumonia_detector, self).__init__()
        self.resnet_model = models.resnet18(pretrained = True, progress = True)
        self.resnet_model.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out = self.resnet_model(x)
        return out

detector_model = pneumonia_detector()

url = 'https://drive.google.com/uc?export=download&id=1dl6NZHd1TfbOlimazzV_DWGkJPfUx-Rq'
model_output = 'pneumonia_detector_model.pth'
#gdown.download(url, model_output, quiet = True)

detector_model.load_state_dict(torch.load(r"weights/pneumonia_detector_model.pth", map_location = torch.device('cpu')))

def diagnose(image):
    image = plt.imread(image) if isinstance(image, str) else np.array(image)
    image = image / 255.
    image.resize((150, 150))
    image = image.reshape(150, 150, 1).repeat(3, axis = -1).reshape(3, 150, 150)
    image = torch.tensor(image).view(1, 3, 150, 150)

    return 'Normal' if detector_model(image.float()).sigmoid() > 0.5 else 'Pneumonia'
