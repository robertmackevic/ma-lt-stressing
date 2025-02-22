import json
import logging
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn import Module

from src.paths import CONFIG_FILE


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(name)
    return logger


def load_config(filepath: Path = CONFIG_FILE) -> Namespace:
    with filepath.open("r") as config:
        return Namespace(**json.load(config))


def save_config(config: Namespace, filepath: Path) -> None:
    with filepath.open("w") as file:
        json.dump(vars(config), file, indent=4)  # type: ignore


def save_weights(filepath: Path, model: Module) -> None:
    torch.save(model.state_dict(), filepath)


def load_weights(filepath: Path, model: Module) -> Module:
    checkpoint = torch.load(filepath, map_location=get_available_device(), weights_only=True)
    model.load_state_dict(checkpoint)
    return model


def get_available_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(module: Module, trainable: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if not trainable or (trainable and p.requires_grad))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
