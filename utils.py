import torchvision.transforms as T
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchviz import make_dot
from PIL import Image
from io import BytesIO
from constants import IMG_SIZE

def RandomPerspectiveAug(image):
    image = torch.tensor(image).view(1, IMG_SIZE, IMG_SIZE)
    return (T.PILToTensor()(T.RandomPerspective(distortion_scale = 0.2, p = 1)(T.ToPILImage()(image))))

def RandomRotation_FlipAug(image):
    image = torch.tensor(image).view(1, IMG_SIZE, IMG_SIZE)
    return (T.PILToTensor()(T.RandomHorizontalFlip(p = 1)(T.RandomRotation(degrees = (0, 30))(T.ToPILImage()(image))) ))

def RandomColorJitterAug(image):
    image = torch.tensor(image).view(1, IMG_SIZE, IMG_SIZE)
    return (T.PILToTensor()(T.ColorJitter(brightness = .65, contrast = .65, saturation = .65)(T.ToPILImage()(image))))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detach_from_gpu(self):
    return self.detach().cpu().numpy()

def visualize_model(model):
    make_dot(0, params=dict(list(model.named_parameters()))).render("model_architecture", format="png")

def numpy_to_bytes(image: np.ndarray):
    plt.imsave('temp_image.jpg', image, cmap = 'autumn_r')
    temp_image = open(r'temp_image.jpg', 'rb').read()
    return temp_image

def bytes_to_numpy(image: bytes):
    image = np.array(Image.open(BytesIO(image)))
    #image = np.expand_dims(image, axis = -1) if image.ndim == 2 else image
    return image

def green_tint(image):
    image = image.transpose(-1, 0, 1)
    image[0] = 0
    image[2] = 0
    return image.transpose(1, 2, 0)

def red_tint(image):
    image = image.transpose(-1, 0, 1)
    image[1] = 0
    image[2] = 0
    return image.transpose(1, 2, 0)
