import torch
from dataset_load import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from torchvision.utils import save_image
from disc_model import Disc
from gen_model import Gen


def train_fn():
    pass


def main():
    disc_H = Disc(in_channels=3).to(config["DEVICE"])
    disc_Z = Disc(in_channels=3).to(config["DEVICE"])
    gen_Z = Gen(img_channels=3, num_residuals=9).to(config["DEVICE"])
    gen_H = Gen(img_channels=3, num_residuals=9).to(config["DEVICE"])
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config["LR"],
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config["LR"],
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config["LOAD_MODEL"]:
        load_checkpoint(
            config["SAVE_DIR_GEN_H"],
            gen_H,
            opt_gen,
            config["LR"],
        )
        load_checkpoint(
            config["SAVE_DIR_GEN_Z"],
            gen_Z,
            opt_gen,
            config["LR"],
        )
        load_checkpoint(
            config["SAVE_DIR_DISC_H"],
            disc_H,
            opt_disc,
            config["LR"],
        )
        load_checkpoint(
            config["SAVE_DIR_DISC_Z"],
            disc_Z,
            opt_disc,
            config["LR"],
        )

    dataset = HorseZebraDataset(
        root_horse=config["TRAIN_DIR"] + "/horses",
        root_zebra=config["TRAIN_DIR"] + "/zebras",
        transform=config.transforms,
    )
    val_dataset = HorseZebraDataset(
        root_horse="cyclegan_test/horse1",
        root_zebra="cyclegan_test/zebra1",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=config["NUM_WORKERS"],
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config["EPOCHS"]):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config["SAVE_MODEL"]:
            save_checkpoint(gen_H, opt_gen, filename=config["SAVE_DIR_GEN_H"])
            save_checkpoint(gen_Z, opt_gen, filename=config["SAVE_DIR_GEN_Z"])
            save_checkpoint(disc_H, opt_disc, filename=config["SAVE_DIR_DISC_H"])
            save_checkpoint(disc_Z, opt_disc, filename=config["SAVE_DIR_DISC_Z"])


if __name__ == "__main__":
    main()
