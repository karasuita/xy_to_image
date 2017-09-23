from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


def load_img(filepath):
    img = ToTensor()(Image.open(filepath).convert('RGB'))
    return img


def save_img(img, filepath):
    ToPILImage()(img).save(filepath)