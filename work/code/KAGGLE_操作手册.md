# Kaggle 操作手册

> 国内网络无法直接访问 HuggingFace，模型需上传到 Kaggle Dataset 后从本地路径加载。

---

## 关键配置

| 项目 | 值 |
|------|-----|
| MODEL_ID | `/kaggle/input/datasets/shizhenhso/metacognition-seed/0ai` |
| Accelerator | GPU T4 x1 |
| Internet | On（安装依赖用，模型从本地加载） |
| gates.json | 需手动 Upload 到 `/kaggle/working/gates.json` |

---

## 运行步骤

### 1. 创建 Notebook

1. 登录 kaggle.com → + Create → New Notebook
2. Settings：
   - Accelerator: **GPU T4 x1**
   - Persistence: **Files only**
   - Internet: **On**
3. 确认左下角显示 **GPU T4 x1**

### 2. 上传 gates.json（必须）

1. Notebook 右侧 File 面板 → Upload
2. 上传项目中的 `work/code/gates.json`
3. 确认文件出现在 `/kaggle/working/gates.json`

> 如果不上传，脚本会报警告但不影响运行。预注册校验会失败。

### 3. 粘贴代码并运行

1. Cell 1：粘贴 runner 脚本全部内容（如 `phase_0_kaggle.py`）
2. Cell 2：输入 `main()` 并运行
3. 等待运行完成

### 4. 下载结果

- `/kaggle/working/results/` 目录下的 JSON 文件
- 右侧 Output 面板 → Download

---

## 各 Phase 运行器

| Phase | 脚本 | 样本数 | 预计时间 |
|-------|------|--------|----------|
| -1 | `work/code/runners/phase_minus_1_kaggle.py` | 200 × 4 tasks | 5-10 min |
| 0 | `work/code/runners/phase_0_kaggle.py` | 1000 × 4 tasks × 3 seeds | 30-50 min |
| 诊断 | `work/code/runners/diagnose_kaggle.py` | 20 × 4 tasks | 1-2 min |

---

## 常见问题

### Q: 模型加载失败
A: 确认 Dataset `shizhenhso/metacognition-seed` 已挂载。路径必须是 `/kaggle/input/datasets/shizhenhso/metacognition-seed/0ai`。

### Q: gates.json 不存在警告
A: 需要手动 Upload `work/code/gates.json` 到 `/kaggle/working/gates.json`。

### Q: CUDA out of memory
A: SmolLM-135M 只占 < 1GB，T4 有 16GB，不太可能。如果出现，加 `torch.cuda.empty_cache()`。

### Q: AUROC < 0.5
A: 信号方向反了。代码已对 entropy 做 `-entropy` 翻转。如果翻转后仍 < 0.5，说明信号基本是噪声。
