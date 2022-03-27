import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image, load_image

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
# layers 0-5-10-19-28
model = models.vgg19(pretrained=True).features

# Hyperparams
config = {
    "total_steps": 6000,
    "lr": 0.001,
    "alpha": 1,
    "beta": 0.01,
}
img = {"galilee": "./styles/galilee.jpg"}

style = {"monalisa": "./styles/monalisa.jpg"}


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features


def load_img(image_name):
    img = Image.open(image_name)
    img_loader = loader(img).unsqueeze(0)
    return img.to(device)


img_size = 356

loader = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


