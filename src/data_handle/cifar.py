import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torchvision import datasets, transforms

from src.utils import seed_worker


@dataclass
class DataLoaderConfig:
    bs: int = 128
    num_workers: int = 4
    prefetch_factor: int = 2
    seed: int = 0
    resume_step: int = 0


@dataclass
class DatasetConfig:
    data_dir: str = "/vol/biomedic3/tx1215/mamo-flow/assets/cifar10/"
    valid_frac: float = 0.05
    split_seed: int = 33
    img_height: int = 32
    img_width: int = 32
    img_channels: int = 3
    use_labels_as_pa: bool = True


class CIFAR10(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        use_labels_as_pa: bool = True,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.use_labels_as_pa = use_labels_as_pa

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Tensor | dict[str, Tensor]]:
        x, y = self.base_dataset[idx]

        pa: dict[str, Tensor] = {}
        if self.use_labels_as_pa:
            y_onehot = torch.zeros(10, dtype=torch.float32)
            y_onehot[int(y)] = 1.0
            pa["y"] = y_onehot

        return {
            "x": x,   # in [0, 1]
            "pa": pa,
        }


def make_cifar_transforms(
    img_height: int,
    img_width: int,
) -> transforms.Compose:
    transform_list = []
    if (img_height, img_width) != (32, 32):
        transform_list.append(transforms.Resize((img_height, img_width)))

    transform_list.extend(
        [
            transforms.ToTensor(),  # [0, 1]
        ]
    )
    return transforms.Compose(transform_list)


def split_train_valid(
    dataset: torch.utils.data.Dataset,
    valid_frac: float,
    split_seed: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if not (0.0 <= valid_frac < 1.0):
        raise ValueError(f"valid_frac must be in [0, 1), got {valid_frac}")

    n_total = len(dataset)
    n_valid = int(round(n_total * valid_frac))
    n_train = n_total - n_valid

    generator = torch.Generator()
    generator.manual_seed(split_seed)

    train_subset, valid_subset = torch.utils.data.random_split(
        dataset,
        [n_train, n_valid],
        generator=generator,
    )
    return train_subset, valid_subset


def get_cifar10(config: DatasetConfig) -> dict[str, CIFAR10]:
    if config.img_channels != 3:
        raise ValueError(
            f"CIFAR-10 expects img_channels=3, got {config.img_channels}"
        )

    transform = make_cifar_transforms(
        config.img_height,
        config.img_width,
    )
    data_dir = Path(config.data_dir)

    full_train = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    train, valid = split_train_valid(
        dataset=full_train,
        valid_frac=config.valid_frac,
        split_seed=config.split_seed,
    )

    datasets_dict = {
        "train": CIFAR10(
            train,
            use_labels_as_pa=config.use_labels_as_pa,
        ),
        "valid": CIFAR10(
            valid,
            use_labels_as_pa=config.use_labels_as_pa,
        ),
        "test": CIFAR10(
            test,
            use_labels_as_pa=config.use_labels_as_pa,
        ),
    }
    return datasets_dict


def get_dataloaders(
    config: DataLoaderConfig,
    datasets_dict: dict[str, torch.utils.data.Dataset],
) -> dict[str, torch.utils.data.DataLoader]:
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_dist:
        from torch.utils.data.distributed import DistributedSampler

    resume_step = config.resume_step
    rank = torch.distributed.get_rank() if is_dist else 0

    dataloaders = {}
    for split in ["train", "valid", "test"]:
        is_train = split == "train"
        seed = int(config.seed + (7654321 * resume_step if is_train else 0))
        sampler = None

        if is_dist:
            sampler = DistributedSampler(
                datasets_dict[split],
                shuffle=is_train,
                seed=seed,
            )

        generator = torch.Generator()
        generator.manual_seed(seed + rank)

        kwargs = dict(
            dataset=datasets_dict[split],
            batch_size=config.bs,
            shuffle=(sampler is None) and is_train,
            drop_last=is_train,
            sampler=sampler,
            pin_memory=True,
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        if config.num_workers > 0:
            kwargs["prefetch_factor"] = config.prefetch_factor
            kwargs["persistent_workers"] = True

        dataloaders[split] = torch.utils.data.DataLoader(**kwargs)

    return dataloaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/vol/biomedic3/tx1215/mamo-flow/assets/cifar10/")
    parser.add_argument("--valid_frac", type=float, default=0.05)
    parser.add_argument("--split_seed", type=int, default=33)
    parser.add_argument("--img_height", type=int, default=32)
    parser.add_argument("--img_width", type=int, default=32)
    parser.add_argument("--img_channels", type=int, default=3)
    parser.add_argument("--use_labels_as_pa", type=int, default=1)

    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset_config = DatasetConfig(
        data_dir=args.data_dir,
        valid_frac=args.valid_frac,
        split_seed=args.split_seed,
        img_height=args.img_height,
        img_width=args.img_width,
        img_channels=args.img_channels,
        use_labels_as_pa=bool(args.use_labels_as_pa),
    )

    dataloader_config = DataLoaderConfig(
        bs=args.bs,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )

    datasets_dict = get_cifar10(dataset_config)
    dataloaders = get_dataloaders(dataloader_config, datasets_dict)

    batch = next(iter(dataloaders["train"]))
    expected_shape = (args.bs, 3, args.img_height, args.img_width)
    assert batch["x"].shape == expected_shape, (
        f"Expected {expected_shape}, got {tuple(batch['x'].shape)}"
    )

    if bool(args.use_labels_as_pa):
        assert "y" in batch["pa"], "Expected pa['y'] to exist"
        assert batch["pa"]["y"].shape == (args.bs, 10), (
            f"Expected pa['y'] shape {(args.bs, 10)}, "
            f"got {tuple(batch['pa']['y'].shape)}"
        )

    x = batch["x"]
    print("Dataset sanity check passed.")
    print(f"Train size: {len(datasets_dict['train'])}")
    print(f"Valid size: {len(datasets_dict['valid'])}")
    print(f"Test size : {len(datasets_dict['test'])}")
    print(f"x range   : [{x.min().item():.3f}, {x.max().item():.3f}]")
    print(f"pa keys   : {list(batch['pa'].keys())}")