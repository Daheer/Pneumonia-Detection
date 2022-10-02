from torch import nn as nn
import torchvision.models as models
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