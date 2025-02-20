import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def get_vgg19():
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg


def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image


def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',  # Fix: Ensure content is stored properly
        '28': 'conv5_1'
    }

    content_features = {}
    style_features = {}

    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            layer_name = layers[name]
            if layer_name == "conv4_1":  # Only this should store content
                content_features['content'] = x
            else:
                style_features[layer_name] = x

    return content_features, style_features


if __name__ == "__main__":
    vgg = get_vgg19()

    content_img = load_image("images/content.jpg")
    style_img = load_image("images/style.jpg")

    content_features, _ = get_features(content_img, vgg)
    _, style_features = get_features(style_img, vgg)

    print("Content Features Extracted:", content_features.keys())
    print("Style Features Extracted:", style_features.keys())
