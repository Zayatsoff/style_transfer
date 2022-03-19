from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset


class AppleOrangeDataset(Dataset):
    def __init__(self, root_orange, root_apple, transform=None):
        self.root_apple = root_apple
        self.root_orange = root_orange
        self.transform = transform

        self.orange_images = os.listdir(root_orange)
        self.apple_images = os.listdir(root_apple)
        self.len_dataset = max(len(self.orange_images), len(self.apple_images))
        self.orange_len = len(self.orange_images)
        self.apple_len = len(self.apple_images)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        orange_img = self.orange_images[index % self.orange_len]
        apple_img = self.apple_images[index % self.apple_len]

        orange_path = os.path.join(self.root_orange, orange_img)
        apple_path = os.path.join(self.root_apple, apple_img)

        orange_img = np.array(Image.open(orange_path).convert("RGB"))
        apple_img = np.array(Image.open(apple_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=orange_img, image0=apple_img)
            orange_img = augmentations["image"]
            apple_img = augmentations["image0"]
        return orange_img, apple_img
