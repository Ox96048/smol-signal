# 元认知种子宣言 — 工程化验证项目

> *"先造尺子，再量高度。尺子本身，最终成为高度的一部分。"*

验证 135M 小模型能否产生可测量的元认知信号，通过 MMQ 框架（AUROC vs entropy baseline）。

## 核心问题

**压缩压力能否触发神经网络的模块化自组织，并产生可测量、可迁移、可自我监控的元认知功能等价物？**

具体而言：是否存在一种信号，使得 ΔMMQ(signal, entropy) > 0 在 OOD 上稳定为正？

## 项目结构

```
├── first10h.txt                    # 10小时执行计划
├── 元认知种子宣言_v5.1.md           # 项目宣言文档
├── requirements.txt                # Python 依赖
├── pytest.ini                      # pytest 配置
├── work/code/
│   ├── common/                     # 通用工具
│   │   ├── seeding.py              # SeedBundle 种子管理
│   │   ├── types.py                # 数据类型定义
│   │   ├── model_loader.py         # 模型加载器
│   │   ├── dataset.py              # 数据集工具
│   │   └── answer_utils.py         # 答案 token 定位
│   ├── signals/                    # 元认知信号计算
│   │   ├── entropy.py              # 熵信号
│   │   ├── margin.py               # top1-top2 logit 差
│   │   ├── signals.py              # max_prob + oracle + SIGNAL_REGISTRY
│   │   ├── registry.py             # 统一入口
│   │   ├── hidden_norm.py          # 隐藏层范数 (Phase 1+)
│   │   ├── self_consistency.py     # 自一致性 (Phase 2+)
│   │   └── verbalized.py           # 语言化置信度 (Phase 3)
│   ├── core/                       # 核心算法
│   │   ├── measure_mmq.py          # MMQ 主测量函数
│   │   ├── stats.py                # Bootstrap AUROC CI
│   │   └── auroc.py                # AUROC + DeLong + 多重比较校正
│   ├── runners/                    # 运行器
│   │   ├── phase_minus_1_kaggle.py # Phase -1 (已完成)
│   │   └── phase_0_kaggle.py       # Phase 0 (下一步)
│   ├── tools/
│   │   └── tokenizer_recon.py      # Tokenizer 侦查
│   ├── gates.json                  # 预注册门控阈值 (冻结)
│   └── D2_SPEC.md                  # D2/D2' 规范 (冻结)
├── work/data/
│   └── generators.py               # 数据生成器 (v2 prompt)
└── history/                        # 归档的旧版文件
```

## 快速开始

### 在 Kaggle 上跑 Phase 0（推荐）

1. 打开 [Kaggle](https://www.kaggle)，新建 Notebook，开启 T4 GPU
2. 把 `work/code/runners/phase_0_kaggle.py` 的内容粘贴到 Cell 1
3. 新建 Cell 2，输入 `main()` 并运行
4. 等待 30-50 分钟，拿到 AUROC + CI 结果

## 核心指标

```
MMQ(M, D, signal_type) = AUROC(signal_type, correctness_on_D)
ΔMMQ(signal, entropy) = MMQ_signal - MMQ_entropy
```

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
