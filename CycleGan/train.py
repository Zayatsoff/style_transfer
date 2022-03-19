import torch
from dataset_load import AppleOrangeDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from config import config, transforms
from torchvision.utils import save_image
from disc_model import Disc
from gen_model import Gen
from tqdm import tqdm


def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    loop = tqdm(loader, leave=True)
    for idx, (orange, apple) in enumerate(loop):
        orange = orange.to(config["DEVICE"])
        apple = apple.to(config["DEVICE"])

        # Train disc H and Z
        with torch.cuda.amp.autocast():
            fake_apple = gen_H(orange)
            D_H_real = disc_H(apple)
            D_H_fake = disc_H(fake_apple.detach())
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.ones_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_orange = gen_Z(apple)
            D_Z_real = disc_Z(orange)
            D_Z_fake = disc_Z(fake_orange.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.ones_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put together
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # train gen H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both gens
            D_H_fake = disc_H(fake_apple)
            D_Z_fake = disc_Z(fake_orange)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_orange = gen_Z(fake_apple)
            cycle_apple = gen_H(fake_orange)
            cycle_orange_loss = l1(orange, cycle_orange)
            cycle_apple_loss = l1(apple, cycle_apple)
            # indentity loss
            # indentity_orange = gen_Z(orange)
            # indentity_apple = gen_H(apple)
            # identity_orange_loss = l1(orange, indentity_orange)
            # identity_apple_loss = l1(apple, indentity_apple)
            # add together
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_orange_loss * config["LAMBDA_CYCLE"]
                + cycle_apple_loss * config["LAMBDA_CYCLE"]
                # + identity_apple_loss * config["LAMBDA_IDENTITY"]
                # + identity_orange_loss * config["LAMBDA_IDENTITY"]
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        if idx % 200 == 0:
            save_image(fake_apple * 0.5 * 0.5, f"saved_image/apple_{idx}.png")
            save_image(fake_orange * 0.5 * 0.5, f"saved_image/orange_{idx}.png")


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

    dataset = AppleOrangeDataset(
        root_apple=config["TRAIN_DIR"] + "/A",
        root_orange=config["TRAIN_DIR"] + "/B",
        transform=transforms,
    )
    val_dataset = AppleOrangeDataset(
        root_apple=config["TEST_DIR"] + "/A",
        root_orange=config["TEST_DIR"] + "/B",
        transform=transforms,
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


# => Saving checkpoint
# Traceback (most recent call last):
#   File "c:/Users/timur/miniconda3/envs/basicpytorch/style_transfer/CycleGan/train.py", line 179, in <module>
#     main()
#   File "c:/Users/timur/miniconda3/envs/basicpytorch/style_transfer/CycleGan/train.py", line 172, in main
#     save_checkpoint(gen_H, opt_gen, filename=config["SAVE_DIR_GEN_H"])
#   File "c:\Users\timur\miniconda3\envs\basicpytorch\style_transfer\CycleGan\utils.py", line 10, in save_checkpoint
#     torch.save(checkpoint, filename)
#   File "C:\Users\timur\miniconda3\envs\basicpytorch\lib\site-packages\torch\serialization.py", line 376, in save
#     with _open_file_like(f, 'wb') as opened_file:
#   File "C:\Users\timur\miniconda3\envs\basicpytorch\lib\site-packages\torch\serialization.py", line 230, in _open_file_like
#     return _open_file(name_or_buffer, mode)
#   File "C:\Users\timur\miniconda3\envs\basicpytorch\lib\site-packages\torch\serialization.py", line 211, in __init__
#     super(_open_file, self).__init__(open(name, mode))
# FileNotFoundError: [Errno 2] No such file or directory: '/saves/gen_h.pth.tar'
# ERROR conda.cli.main_run:execute(33): Subprocess for 'conda run ['python', 'c:/Users/timur/miniconda3/envs/basicpytorch/style_transfer/CycleGan/train.py']' command failed.  (See above for error)
