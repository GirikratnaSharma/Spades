import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# Load VGG-19 Model
def get_vgg19():
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    for param in vgg.parameters():
        param.requires_grad = False  # Freeze weights
    return vgg


# Load Image and Transform
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Compute Gram Matrix for Style Loss
def gram_matrix(tensor):
    B, C, H, W = tensor.shape  # Fix: Ensure input is 4D
    tensor = tensor.view(C, H * W)  # Flatten feature map
    return torch.mm(tensor, tensor.t())  # Compute Gram matrix


# Extract Content and Style Features
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',  # Content Layer
        '28': 'conv5_1'
    }

    content_features = {}
    style_features = {}

    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            layer_name = layers[name]
            if layer_name == "conv4_1":
                content_features['content'] = x
            else:
                style_features[layer_name] = x  # ✅ Keep as feature map, compute Gram matrix later

    return content_features, style_features


# Define Content Loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        return torch.mean((input - self.target) ** 2)


# Define Style Loss
class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()  # ✅ Compute Gram Matrix here

    def forward(self, input):
        gram_input = gram_matrix(input)  # ✅ Compute Gram Matrix here
        return torch.mean((gram_input - self.target) ** 2)


# Neural Style Transfer Function
def style_transfer(content_path, style_path, num_steps=500, alpha=1, beta=1e6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = get_vgg19().to(device).eval()

    # Load images
    content_img = load_image(content_path).to(device)
    style_img = load_image(style_path).to(device)

    # Extract Features
    content_features, _ = get_features(content_img, vgg)
    _, style_features = get_features(style_img, vgg)

    # Initialize Generated Image (Copy of Content Image)
    generated = content_img.clone().requires_grad_(True).to(device)

    # Define Losses
    content_loss = ContentLoss(content_features['content'])
    style_losses = {layer: StyleLoss(style_features[layer]) for layer in style_features}

    # Optimizer
    optimizer = optim.Adam([generated], lr=0.01)

    for step in range(num_steps):
        optimizer.zero_grad()
        gen_content_features, gen_style_features = get_features(generated, vgg)

        # Compute Losses
        c_loss = content_loss(gen_content_features['content'])
        s_loss = sum(style_losses[layer](gen_style_features[layer]) for layer in style_losses)  # ✅ Style loss now works

        # Total Loss
        total_loss = alpha * c_loss + beta * s_loss
        total_loss.backward()
        optimizer.step()

        # Print Progress
        if step % 50 == 0:
            print(f"Step {step}: Content Loss: {c_loss.item()}, Style Loss: {s_loss.item()}")

    # Convert to PIL Image
    final_image = generated.cpu().detach().squeeze(0)
    final_image = transforms.ToPILImage()(final_image)
    final_image.show()

    return final_image


# Run NST
if __name__ == "__main__":
    output = style_transfer("images/content.jpg", "images/style.jpg", num_steps=1000)
    output.save("images/output.jpg")
