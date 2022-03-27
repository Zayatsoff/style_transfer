import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# changed: - all 16 layerss of vgg19

# Hyperparams
config = {
    "total_steps": 6000,
    "lr": 0.001,
    "alpha": 1,
    "beta": 0.01,
}
img = {"img0": "./styles/galilee.jpg"}
style = {"style0": "./styles/monalisa.jpg"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.chosen_features = [
            "2",
            "5",
            "9",
            "12",
            "16",
            "19",
            "22",
            "25",
            "29",
            "32",
            "35",
            "38",
            "42",
            "45",
            "48",
            "51",
        ]
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
    img = loader(img).unsqueeze(0)
    return img.to(device)


def content_loss(gen_features, c_features, style_features, c_loss):
    for (
        gen_feature,
        c_feature,
        style_feature,
    ) in zip(gen_features, c_features, style_features):
        batch_size, channel, height, width = gen_feature.shape
        c_loss += torch.mean((gen_feature - c_feature) ** 2)
        return c_loss


def style_loss(gen_features, og_img_features, style_features, style_loss):
    for (
        gen_feature,
        og_feature,
        style_feature,
    ) in zip(gen_features, og_img_features, style_features):
        batch_size, channel, height, width = gen_feature.shape
        # compute gram matrix
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )

        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        style_loss += torch.mean((G - A) ** 2)
        return style_loss


img_size = 356

loader = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

invTrans = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)


# freeze the weights
model = VGG19().to(device).eval()
og_img = load_img(img["img0"])
style_img = load_img(style["style0"])
generated = og_img.clone().requires_grad_(True)
# generated = torch.randn(load_img(img["img0"]).shape, device=device, requires_grad=True)

optimizer = optim.Adam([generated], lr=config["lr"])
for step in range(config["total_steps"]):
    gen_features = model(generated)
    c_features = model(og_img)
    s_features = model(style_img)

    s_loss = c_loss = 0
    c_loss = content_loss(gen_features, c_features, s_features, c_loss)
    s_loss = style_loss(gen_features, c_features, s_features, s_loss)
    total_loss = config["alpha"] * c_loss + config["beta"] * s_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print("total loss : ", total_loss)
        save_gen = invTrans(generated)
        save_image(save_gen, f"generated_{step}.png")
