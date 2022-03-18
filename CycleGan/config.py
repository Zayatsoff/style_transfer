import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

config = {
    "BATCH_SIZE": 1,
    "LR": 2e-4,
    "SAVE_DIR_GEN_H": "./saves/gen_h.pth.tar",
    "SAVE_DIR_GEN_Z": "./saves/gen_z.pth.tar",
    "SAVE_DIR_DISC_H": "./saves/disc_h.pth.tar",
    "SAVE_DIR_DISC_Z": "./saves/disc_z.pth.tar",
    "DEVICE": "cuda",
    "LAMBDA_IDENTITY": 0.0,
    "LAMBDA_CYCLE": 10,
    "NUM_WORKERS": 4,
    "EPOCHS": 200,
    "LOAD_MODEL": False,
    "SAVE_MODEL": True,
    "TRAIN_DIR": "./data/train",
    "TEST_DIR": "./data/test",
}

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
