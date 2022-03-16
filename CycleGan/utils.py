import torch


def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # set old lr to new lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
