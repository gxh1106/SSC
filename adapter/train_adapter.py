import os
import torch
import sys
import datetime
import config
import random
import numpy as np
import torchvision
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
from network_adapter import network
from logger import get_logger
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets import Vimeo, Div2K, CIFAR10Dataset
current_path = Path(__file__).resolve().parents[0]
if str(current_path) not in sys.path:
    sys.path.append(str(current_path))
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train adapter model from scratch")
    parser.add_argument("--DDP", default=False, action="store_true", help="Enable DistributedDataParallel (torchrun)")
    parser.add_argument("--local_rank", default=int(os.getenv("LOCAL_RANK", -1)), type=int, help="Local rank for DDP (set by torchrun)")
    parser.add_argument("--tag", default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--save_path", default=str(current_path / "outputs"))
    parser.add_argument("--data_path", default=str(current_path / "datasets"))
    parser.add_argument("--dataset", choices=["div2k", "cifar10", "vimeo"], default="div2k", help="choose training dataset")
    # Reserved for future use
    parser.add_argument("--load_path", default=None)

    args = parser.parse_args()
    # Normalize path-like args to Path
    args.save_path = Path(args.save_path)
    args.data_path = Path(args.data_path)
    return args

if __name__ == "__main__":
    args = parse_args()

    # Ensure output dirs exist before creating logger
    args.save_path.mkdir(parents=True, exist_ok=True)
    (args.save_path / "vision").mkdir(parents=True, exist_ok=True)
    logger = get_logger("train", args.save_path / f"train_{args.tag}.log")

    # Seeding
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    # Device and (optional) DDP setup
    use_ddp = bool(args.DDP) or int(os.getenv("WORLD_SIZE", "1")) > 1
    if use_ddp:
        # Expect launched by: torchrun --nproc_per_node=... train_adapter.py --DDP
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    # Build model
    model = network(config).to(device)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=float(config.lr), amsgrad=True)

    # Dataset & DataLoader
    # Choose dataset
    if args.dataset == "div2k":
        train_dataset = Div2K(args.data_path, if_train=True, crop_size=256)
    elif args.dataset == "cifar10":
        train_dataset = CIFAR10Dataset(args.data_path, if_train=True, download=True)
    else:
        train_dataset = Vimeo(args.data_path)
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        sampler=sampler,
        num_workers=getattr(config, "num_workers", 4),
        drop_last=True,
    )

    # Train
    model.train()
    for epoch in range(config.epochs):
        if use_ddp:
            sampler.set_epoch(epoch)

        if rank == 0:
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        else:
            train_bar = train_loader

        running_loss, running_psnr = 0.0, 0.0
        num = 0

        for data in train_bar:
            num += 1
            optim.zero_grad(set_to_none=True)

            data = data.to(device, non_blocking=True)
            pack = model(data)
            x_hat = pack["x_hat"]

            recon_loss = torch.mean((x_hat - data) ** 2)
            loss = recon_loss + pack["kl_loss"]

            loss.backward()
            # Optional gradient clip
            if getattr(config, "gradient_clip", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optim.step()

            with torch.no_grad():
                mse = torch.mean((data - x_hat) ** 2)
                psnr = torch.mean(10 * torch.log10(1.0 / mse))
                running_loss += float(loss.detach().cpu())
                running_psnr += float(psnr.detach().cpu())

            if rank == 0 and num % getattr(config, "log_interval", 100) == 0:
                logger.info(f"iter {num}: loss={running_loss / num:.6f}, psnr={running_psnr / num:.3f}")

        # Save checkpoint and sample image (rank 0 only)
        if rank == 0:
            # DDP wraps model; get underlying module when needed
            to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(to_save, args.save_path / f"baseline_{args.tag}_e{epoch+1}.pt")
            # Save the last batch reconstruction preview (map back to [0,1])
            try:
                preview = (x_hat.detach().clamp(-1, 1) + 1) / 2.0
                save_image(preview, args.save_path / "vision" / f"{epoch+1}.png")
            except Exception:
                pass

    # Cleanup
    if use_ddp:
        dist.destroy_process_group()


            
