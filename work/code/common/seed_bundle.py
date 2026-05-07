"""SeedBundle: 统一管理四个独立随机源.

建议与批评 §8 指出:
  - data_seed / sampling_seed / probe_init_seed / spl_init_seed 是四个独立随机源
  - seed=42 时这四个都=42 vs seed分别=42/43/44/45, 方差完全不同
  - 代码里永远传 SeedBundle, 不传裸整数
"""

import random
import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SeedBundle:
    """四个独立随机源的种子捆绑.

    构造方式:
      SeedBundle(base=42)           -> 四个 seed 自动派生 (推荐)
      SeedBundle(data=42, samp=43, probe=44, spl=45)  -> 手动指定
    """
    data_seed: int = 42
    sampling_seed: int = 43
    probe_init_seed: int = 44
    spl_init_seed: int = 45

    @classmethod
    def from_base(cls, base: int = 42) -> "SeedBundle":
        """从单一 base seed 派生四个独立 seed.

        用 hash 确保不同 base seed 产生完全不同的四个 seed,
        避免 seed=42 和 seed=43 的四个 seed 之间有任何模式.
        """
        def derive(base: int, label: str) -> int:
            digest = hashlib.md5(f"{base}:{label}".encode()).hexdigest()
            return int(digest[:8], 16) % (2 ** 31)

        return cls(
            data_seed=derive(base, "data"),
            sampling_seed=derive(base, "sampling"),
            probe_init_seed=derive(base, "probe"),
            spl_init_seed=derive(base, "spl"),
        )

    def with_overrides(
        self,
        data: Optional[int] = None,
        sampling: Optional[int] = None,
        probe: Optional[int] = None,
        spl: Optional[int] = None,
    ) -> "SeedBundle":
        """返回部分覆盖的 SeedBundle (不变的部分用原值)."""
        return SeedBundle(
            data_seed=data if data is not None else self.data_seed,
            sampling_seed=sampling if sampling is not None else self.sampling_seed,
            probe_init_seed=probe if probe is not None else self.probe_init_seed,
            spl_init_seed=spl if spl is not None else self.spl_init_seed,
        )

    def set_all(self):
        """设置 Python/NumPy/PyTorch 所有随机状态.

        注意: 只设 data_seed 和 sampling_seed 到全局状态.
        probe_init_seed 和 spl_init_seed 应在初始化 probe/SPL 头时
        局部传入, 不污染全局随机状态.
        """
        import numpy as np
        import torch
        random.seed(self.data_seed)
        np.random.seed(self.data_seed)
        torch.manual_seed(self.data_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.data_seed)
