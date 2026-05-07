"""统一管理所有随机源。永远传 bundle，永远不传裸整数。"""
import random
import numpy as np
import torch
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SeedBundle:
    data_seed: int = 42
    sampling_seed: int = 43
    probe_init_seed: int = 44
    spl_init_seed: int = 45

    def apply_global(self):
        """跑主实验前调用。注意：不包含 sampling_seed，那个每次采样单独用。"""
        random.seed(self.data_seed)
        np.random.seed(self.data_seed)
        torch.manual_seed(self.data_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.data_seed)

    def replace(self, **kwargs):
        """生成一个新 bundle（不可变）"""
        return replace(self, **kwargs)


DEFAULT_BUNDLE = SeedBundle()
