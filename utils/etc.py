import os
import torch
import numpy as np
import random
import yaml
from typing import Union, List, Tuple, Dict
from omegaconf import OmegaConf, DictConfig, ListConfig


def apply_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def resolve_thread_count(n_threads, cfg=None) -> int:
    count = max(int(n_threads), 1) if n_threads is not None else 1
    if cfg is None:
        return count
    try:
        cpu_cap = int(cfg.resource.num_cpus) - 1
    except Exception:
        return count
    return min(count, cpu_cap) if cpu_cap > 0 else count


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def get(self):
        return self.sum / self.count


class FlowStyleListDumper(yaml.Dumper):
    def represent_sequence(self, tag, sequence, flow_style=True):
        return super().represent_sequence(tag, sequence, flow_style)


def dump_cfg(cfg: Union[dict, DictConfig, ListConfig], stream=None):
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg)
    return yaml.dump(cfg, stream, Dumper=FlowStyleListDumper, sort_keys=False)


def flatten_list(list_obj: list) -> list:

    list_flat = []
    for list_in in list_obj:
        list_flat += list_in
    return list_flat


def list_to_tuple(batch: List[tuple]) -> Tuple[Union[list, dict]]:
    """
    Arguments:
        batch = List[tuple1, tuple2, ...]
    Returns:
        batch = tuple(List1 or Dict1(list), List2 or Dict2(list), ...)
    """

    batch = list(zip(*batch))

    for idx, samples in enumerate(batch):
        if isinstance(samples[0], dict):
            samples_ = {}
            for sample in samples:
                for key, value in sample.items():
                    if key in samples_:
                        samples_[key].append(value)
                    else:
                        samples_[key] = [value]
            batch[idx] = samples_

    return tuple(batch)


def convert_tuple_to_tensor(
    batch: Tuple[Union[list, dict]], device: str
) -> Tuple[torch.Tensor]:
    """
    Arguments:
        batch = tuple(List1 or Dict1(list), List2 or Dict2(list), ...)
    Returns:
        batch = tuple(torch.tensor1 or Dict1(torch.tensor), torch.tensor2 or Dict2(torch.tensor), ...)
    """

    batch = list(batch)
    for idx, samples in enumerate(batch):
        if isinstance(samples, dict):
            for key, value in samples.items():
                samples[key] = torch.tensor(value).to(device)
                if samples[key].dtype == torch.float64:
                    samples[key] = samples[key].to(torch.float32)
            batch[idx] = samples
        else:
            batch[idx] = torch.tensor(samples).to(device)
            if batch[idx].dtype == torch.float64 or batch[idx].dtype == torch.bool:
                batch[idx] = batch[idx].to(torch.float32)
            if batch[idx].ndim == 1:
                batch[idx] = batch[idx].unsqueeze(1)

    return tuple(batch)


def get_unique_dir(base_dir, prefix):
    """
    base_dir: 상위 디렉토리 (예: ".data/field_pcd")
    prefix: 날짜 문자열 (예: "260331")
    """
    counter = 1
    while True:
        dir_name = f"{prefix}_{counter}"
        full_path = os.path.join(base_dir, dir_name)

        if not os.path.exists(full_path):
            return full_path

        counter += 1


class SuppressOutput:
    def __enter__(self):
        self._stdout_fd = os.dup(1)
        self._stderr_fd = os.dup(2)

        self._null = os.open(os.devnull, os.O_RDWR)
        os.dup2(self._null, 1)
        os.dup2(self._null, 2)

    def __exit__(self, *args):
        os.dup2(self._stdout_fd, 1)
        os.dup2(self._stderr_fd, 2)
        os.close(self._null)
        os.close(self._stdout_fd)
        os.close(self._stderr_fd)
