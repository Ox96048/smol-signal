# 数据集清单

## 概述
本文件记录项目中所有数据集的元信息。

## 数据集列表

### arithmetic_easy.json
- **类型**: 简单加法
- **描述**: 1-10 范围的加法题目
- **样本数**: 200
- **生成器**: generators/arithmetic_easy.py
- **生成时间**: 2026-05-06
- **Checksum**: TBD

### arithmetic_hard.json
- **类型**: 两位数加减
- **描述**: 10-99 范围的加法和减法题目
- **样本数**: 200
- **生成器**: generators/arithmetic_hard.py
- **生成时间**: 2026-05-06
- **Checksum**: TBD

### pattern_completion.json
- **类型**: 模式补全
- **描述**: 数字序列模式识别任务
- **样本数**: 200
- **生成器**: generators/pattern_completion.py
- **生成时间**: 2026-05-06
- **Checksum**: TBD

### simple_logic.json
- **类型**: 简单逻辑
- **描述**: 常识逻辑填空题目
- **样本数**: 200
- **生成器**: generators/simple_logic.py
- **生成时间**: 2026-05-06
- **Checksum**: TBD

### d2_constructor.json
- **类型**: D2 构造器（同词不同理）
- **描述**: 类别判断任务，测试概念理解
- **样本数**: 500
- **生成器**: generators/d2_constructor.py
- **生成时间**: 2026-05-06
- **Checksum**: TBD

### d2_prime_constructor.json
- **类型**: D2' 构造器（同词同理）
- **描述**: 数学运算任务，换数字保持逻辑
- **样本数**: 500
- **生成器**: generators/d2_prime_constructor.py
- **生成时间**: 2026-05-06
- **Checksum**: TBD

## 数据集分类

| 类别 | 数据集 | 样本数 |
|------|--------|--------|
| Phase -1 | arithmetic_easy.json | 200 |
| Phase -1 | arithmetic_hard.json | 200 |
| Phase -1 | pattern_completion.json | 200 |
| Phase -1 | simple_logic.json | 200 |
| Phase 0 | d2_constructor.json | 500 |
| Phase 0 | d2_prime_constructor.json | 500 |

## 数据格式
所有数据集采用统一的 JSON 格式：
```json
{
  "id": "unique_identifier",
  "prompt": "问题文本",
  "answer": "正确答案",
  "meta": {
    "type": "数据集类型",
    ...其他元信息
  }
}
```