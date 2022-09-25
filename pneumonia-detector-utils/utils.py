import torchvision.transforms as T
import torch

def RandomPerspectiveAug(image):
    image = torch.tensor(image).view(1, 150, 150)
    return (T.PILToTensor()(T.RandomPerspective(distortion_scale = 0.2, p = 1)(T.ToPILImage()(image))))

def RandomRotation_FlipAug(image):
    image = torch.tensor(image).view(1, 150, 150)
    return (T.PILToTensor()(T.RandomHorizontalFlip(p = 1)(T.RandomRotation(degrees = (0, 30))(T.ToPILImage()(image))) ))

def RandomColorJitterAug(image):
    image = torch.tensor(image).view(1, 150, 150)
    return (T.PILToTensor()(T.ColorJitter(brightness = .65, contrast = .65, saturation = .65)(T.ToPILImage()(image))))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detach_from_gpu(self):
    return self.detach().cpu().numpy()

