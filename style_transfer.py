import torch
import torch.nn as nn
import torchvision.models as models

def get_vgg19():
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

if __name__ == "__main__":
    vgg = get_vgg19()
    print(vgg)
