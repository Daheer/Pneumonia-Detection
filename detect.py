import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.image as mat_img
import matplotlib.pyplot as plt
import numpy as np

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

detector_model.load_state_dict(torch.load(r"C:\Users\Abdul Karim\Desktop\Pneumonia-Detection\weights\pneumonia_detector_model.pth", map_location = torch.device('cpu')))

test_image = mat_img.imread('test-image.jpeg')
test_image = np.array(test_image) / 255.
test_image.resize((150, 150))
test_image = test_image.reshape(150, 150, 1).repeat(3, axis = -1).reshape(3, 150, 150)
test_image = torch.tensor(test_image).view(1, 3, 150, 150)

print('Normal' if detector_model(test_image.float()).sigmoid() > 0.5 else 'Pneumonia')