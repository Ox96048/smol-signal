# 元认知种子宣言 — 工程化验证项目

> *"先造尺子，再量高度。尺子本身，最终成为高度的一部分。"*

验证 135M 小模型能否产生可测量的元认知信号，通过 MMQ 框架（AUROC vs entropy baseline）。

## 核心问题

**压缩压力能否触发神经网络的模块化自组织，并产生可测量、可迁移、可自我监控的元认知功能等价物？**

具体而言：是否存在一种信号，使得 ΔMMQ(signal, entropy) > 0 在 OOD 上稳定为正？

## 项目结构

```
├── 元认知种子宣言_v5.0.md          # 项目核心文档（哲学+路线）
├── requirements.txt                # Python 依赖
├── pytest.ini                      # pytest 配置
├── work/code/
│   ├── common/                     # 通用工具
│   │   ├── types.py                # 数据类型定义
│   │   ├── seeding.py              # SeedBundle 种子管理
│   │   ├── model_loader.py         # 模型加载器
│   │   └── dataset.py              # 数据集工具
│   ├── signals/                    # 元认知信号计算
│   │   ├── entropy.py              # 熵信号
│   │   ├── margin.py               # top1-top2 logit 差
│   │   ├── max_logit.py            # max_prob (softmax 概率)
│   │   ├── hidden_norm.py          # 隐藏层范数
│   │   ├── self_consistency.py     # 自一致性
│   │   ├── verbalized.py           # 语言化置信度
│   │   ├── signals.py              # 信号注册 + oracle 信号
│   │   └── registry.py             # 统一入口
│   ├── core/                       # 核心算法
│   │   ├── measure_mmq.py          # MMQ 主测量函数
│   │   ├── bootstrap_ci.py         # 通用 bootstrap CI
│   │   └── stats.py                # AUROC 专用 bootstrap CI
│   ├── runners/                    # 运行器
│   │   ├── phase_minus_1.py        # Phase -1 运行器
│   │   └── phase_minus_1_kaggle.py # Kaggle 自包含版
│   ├── tests/                      # 测试
│   ├── tools/
│   │   └── tokenizer_recon.py      # Tokenizer 侦查
│   ├── gates.json                  # 预注册门控阈值
│   ├── D2_SPEC.md                  # D2/D2' 规范冻结
│   └── KAGGLE_操作手册.md           # Kaggle 操作步骤
└── work/data/
    └── generators/                 # 数据生成器
```

## 快速开始

### 本地测试

```bash
pip install -r requirements.txt
pytest
```

### 在 Kaggle 上跑 Phase -1（推荐）

1. 打开 [Kaggle](https://www.kaggle.com)，新建 Notebook，开启 T4 GPU + Internet
2. 把 `work/code/runners/phase_minus_1_kaggle.py` 的内容粘贴到 Cell 1
3. 新建 Cell 2，输入 `main()` 并运行
4. 等待 5-10 分钟，拿到第一批真实 AUROC 数字

详见 [KAGGLE_操作手册.md](work/code/KAGGLE_操作手册.md)

## 核心指标

```
MMQ(M, D, signal_type) = AUROC(signal_type, correctness_on_D)
ΔMMQ(signal, entropy) = MMQ_signal - MMQ_entropy
```

**判决核心**：是否存在一种 signal_type，使得 ΔMMQ > 0 在 OOD 上稳定为正，且 bootstrap CI 不包含 0？

## GATE 决策体系

| GATE | 阶段 | 判决标准 |
|------|------|---------|
| -1 | 可行性底线 | accuracy ∈ [0.40, 0.85] |
| 0 | 仪器稳定性 | CI width ≤ 0.05 |
| 1 | 免费午餐基线 | B ≥ 0.55 |
| 2 | 监督增量 | C > B + 0.03 |
| 3 | 泛化真伪 | OOD 上 ΔMMQ > 0 |

阈值已预注册在 [gates.json](work/code/gates.json)，不允许事后修改。

## 许可证

MIT License
