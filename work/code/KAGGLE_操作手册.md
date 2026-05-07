# Kaggle 操作手册：从零到 Phase -1 结果

> 目标：一个下午拿到 SmolLM-135M 的第一批真实 AUROC 数字。
> 预计总耗时：30 分钟设置 + 10 分钟运行。

---

## 第一步：注册 Kaggle 账号（5 分钟）

1. 打开 https://www.kaggle.com
2. 点右上角 **Register**，用 Google 邮箱注册
3. 注册完成后，进入 https://www.kaggle.com/settings/account
4. 往下滚动找到 **API** 区域
5. 点 **Create New Token**，会下载一个 `kaggle.json` 文件
   - 这个文件包含你的用户名和 API Key
   - **不要分享给任何人**
   - 留着备用（如果以后要用命令行操作的话）

> 如果你已经有 Kaggle 账号，跳过这步。

---

## 第二步：创建 Notebook 并开启 GPU（3 分钟）

1. 登录 Kaggle 后，点右上角 **+ Create** → **New Notebook**
2. 进入 Notebook 编辑界面后，看右侧边栏（如果没看到，点右上角齿轮图标）
3. **Settings 面板**设置如下：

| 设置项 | 选择 | 为什么 |
|--------|------|--------|
| Accelerator | **GPU T4 x1** | SmolLM-135M 需要 GPU 才跑得快 |
| Persistence | **Files only** | 代码重启会丢，但输出文件保留 |
| Internet | **On** | 要从 HuggingFace 下载模型 |
| Language | **Python** | 不用改 |

4. 确认左下角显示 **GPU T4 x1**（不是 CPU，不是 P100）

> ⚠️ 如果 Accelerator 选项灰色不可选，说明你的账号还没验证手机号。去 Settings → Phone Verification 验证。

---

## 第三步：上传 gates.json（1 分钟）

1. 在 Notebook 右侧 **File** 面板，点 **Upload**
2. 上传项目中的 `work/code/gates.json` 文件
3. 确认文件出现在 `/kaggle/working/gates.json`

> 这步不是必须的，但 first10h 方案要求预注册门控文件存在。不上传也不影响运行。

---

## 第四步：粘贴并运行代码（2 分钟粘贴 + 10 分钟运行）

### Cell 1：粘贴全部代码

1. 在 Notebook 的第一个 Cell 中，**删掉默认内容**
2. 打开项目文件 `work/code/runners/phase_minus_1_kaggle.py`
3. **全选复制**，粘贴到 Cell 1 中
4. 点 **Run**（或按 Shift+Enter）

> 这个 Cell 只是定义函数，不会跑实验。应该瞬间完成，没有输出。

### Cell 2：运行主函数

1. 在 Cell 1 下方新建一个 Cell
2. 输入以下内容：

```python
main()
```

3. 点 **Run**（或按 Shift+Enter）
4. **等待 5-10 分钟**

### 你会看到的输出顺序

```
==================================================
H0-1: Tokenizer 侦查（必须先跑，决定后续一切）
==================================================
单 token 答案数: XX
...（tokenizer 分析输出）...
✅ 单 token 答案有 XX 个，足够继续。

==================================================
运行单元测试...
==================================================
  ✅ test_entropy_uniform: H=4.6052, expected=4.6052
  ✅ test_entropy_peaked: H=0.000000
  ✅ test_entropy_two_peaks: H=0.6931, expected=0.6931
  ✅ test_entropy_1d_assert: 2D logits correctly rejected
  ✅ test_margin: 7.0
  ✅ test_max_prob: 0.999999
  ✅ test_max_prob_range: 0.0037 ∈ [0, 1]
  ✅ test_neg_log_prob_correct: 0.0067
  ✅ test_bootstrap_perfect: AUROC=0.998, CI=[0.995, 1.000]
  ✅ test_bootstrap_random: AUROC=0.503, CI=[0.453, 0.554]
  ✅ test_bootstrap_degenerate: labels are all same class

🎉 所有 11 个测试通过！可以继续跑实验。

==================================================
加载模型
==================================================
Loaded in XX.Xs
Model params: 135.0M
Device: cuda

=== Task: add_small ===
  原始 200 样本，保留 XXX 个单 token 答案 (丢弃率 XX.X%)
  50/XXX done (XX.Xs)
  100/XXX done (XX.Xs)
  ...
  accuracy: 0.XXX
  entropy: AUROC=0.XXX [0.XXX, 0.XXX] width=0.XXX
  margin: AUROC=0.XXX [0.XXX, 0.XXX] width=0.XXX
  max_prob: AUROC=0.XXX [0.XXX, 0.XXX] width=0.XXX
  neg_log_prob_correct: AUROC=0.XXX [0.XXX, 0.XXX] width=0.XXX

（重复 4 个任务...）

============================================================
GATE -1 决策汇总
============================================================
  add_small: ✅ acc=0.XX ∈ [0.40, 0.85], entropy AUROC=0.XXX
  sub_small: ✅ acc=0.XX ∈ [0.40, 0.85], entropy AUROC=0.XXX
  compare: ⚠️ acc=0.XX 太低（模型无法胜任）
  parity: ⚠️ acc=0.XX 太高（信号饱和）

通过 GATE -1 的任务: ['add_small', 'sub_small']
🟢 2 个任务通过 → 可进入 Phase 0

结果已保存到 /kaggle/working/results/phase_minus_1_mini.json
Tokenizer 侦查已保存到 /kaggle/working/results/tokenizer_recon.json
```

---

## 第五步：下载结果（2 分钟）

1. 等 Cell 2 跑完（看到"GATE -1 决策汇总"和最终输出）
2. 在 Notebook 右侧 **Output** 面板，点 **Download** 下载所有输出文件
3. 或者手动下载关键文件：
   - `/kaggle/working/results/phase_minus_1_mini.json` ← **最重要的结果**
   - `/kaggle/working/results/tokenizer_recon.json` ← tokenizer 分析

### 手动下载方式

1. 在 Cell 3 中运行：

```python
import os
for root, dirs, files in os.walk('/kaggle/working/results'):
    for f in files:
        print(os.path.join(root, f))
```

2. 然后在右侧 File 面板找到这些文件，点文件右侧的下载图标

---

## 第六步：解读结果并决策（5 分钟）

打开下载的 `phase_minus_1_mini.json`，对照下表：

### GATE -1 判断标准

每个任务看两个数字：**accuracy** 和 **entropy AUROC**

| accuracy 范围 | 含义 | 判定 |
|---------------|------|------|
| 0.40 - 0.85 | 模型有对有错，信号有区分空间 | ✅ 通过 |
| < 0.40 | 模型基本全错，无法区分 | ⚠️ 太难 |
| > 0.85 | 模型基本全对，信号饱和 | ⚠️ 太容易 |

### 四种结局和下一步

| 结局 | 症状 | 下一步 |
|------|------|--------|
| 🟢 **通过** | ≥ 2 个任务 accuracy ∈ [0.40, 0.85] | 写 Phase 0 全量版（1000样本 × 全部signal） |
| 🟡 **勉强通过** | 只有 1 个任务在区间内 | 用这一个任务做 Phase 0，论文里诚实说明 |
| 🟡 **全部太容易** | 所有 accuracy > 0.85 | 升级任务难度：两位数运算、三元素比较 |
| 🔴 **全部太难** | 所有 accuracy < 0.40 | 72h 内决定：切 SmolLM-360M 或降任务难度 |
| 🔴 **信号全随机** | accuracy 合理但所有 AUROC ≈ 0.5 | 数据量太小或任务没有难易梯度 |

### 关键信号解读

| 信号 | 含义 | 期望 |
|------|------|------|
| **entropy AUROC** | 熵能否区分对/错 | > 0.55 说明有信号 |
| **margin AUROC** | top1-top2 差能否区分 | 通常比 entropy 稍好 |
| **max_prob AUROC** | 最大概率能否区分 | 和 margin 高度相关 |
| **neg_log_prob_correct AUROC** | oracle 信号（知道答案时） | **理论上限**，其他信号不应超过这个 |

---

## 常见问题

### Q: 运行时报 CUDA out of memory
A: 不太可能，SmolLM-135M 只占 < 1GB 显存，T4 有 16GB。如果真的出现，在代码开头加：
```python
torch.cuda.empty_cache()
```

### Q: 模型下载很慢
A: 确保 Internet 设置为 **On**。HuggingFace 下载 SmolLM-135M 约需 1-3 分钟。如果超时，重跑一次 Cell 2。

### Q: 某个任务 accuracy = 0.0 或 1.0
A: 正常。135M 模型在某些任务上可能全对或全错。只要 ≥ 1 个任务在 [0.40, 0.85] 区间内就算通过。

### Q: AUROC 显示 < 0.5
A: 说明信号方向反了（entropy 越高=越不确定=越可能错）。代码已经用 `-entropy` 做了翻转，如果还是 < 0.5 说明信号基本是噪声。

### Q: 想用 API 命令行操作而不是网页
A: 可以。安装 `pip install kaggle`，配置 `~/.kaggle/kaggle.json`，然后用：
```bash
# 初始化 notebook 元数据
kaggle kernels init -p ./kaggle_notebook/

# 编辑 kernel-metadata.json，设置 enable_gpu: true

# 上传并执行
kaggle kernels push -p ./kaggle_notebook/

# 查看状态
kaggle kernels status 你的用户名/notebook-name

# 下载输出
kaggle kernels output 你的用户名/notebook-name -p ./results/
```

---

## 执行后必做的三件事

1. **把 `/kaggle/working` 整个下载到本地**，做一次 git commit。这是第一个真实 checkpoint。

2. **填完 RUNLOG.md**。在 `work/code/RUNLOG.md` 中勾选完成的 checkbox，诚实写卡壳记录。

3. **根据 GATE -1 结果决定下一步**：
   - 通过 → 告诉我结果数字，我给你 Phase 0 全量版代码
   - 没通过 → 告诉我具体症状，我们根据数字决定 pivot 方向
