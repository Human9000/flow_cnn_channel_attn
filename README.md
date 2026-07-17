# Flow CNN Channel Attention

基于通道组竞争-互补耦合的 CNN 注意力机制与流式推理框架。

## 概览

本项目包含三个核心部分：

1. **竞争-互补耦合模块** — 替代标准 1×1 卷积的通道分组多头注意力机制
2. **流式推理框架** — 确保逐块流式推理与整段推理数值严格一致的 ONNX 转换与 C 运行时
3. **粒化 SE 模块 (GSE)** — 基于空间下采样的粒化压缩-激励注意力

## 项目结构

```
├── channel_attn/                # 竞争-互补注意力模块
│   ├── group_mha.py             #   核心实现 (MultiHeadAttention + GroupMHA)
│   ├── theory.md                #   理论推导与设计文档
│   └── gse.py                   #   粒化压缩-激励模块 (GranularSE)
├── streaming/                   # 流式推理框架
│   ├── design.md                #   数值一致性方案
│   ├── models/
│   │   ├── resnet50.py          #   ResNet50 模型定义
│   │   └── unet.py              #   UNet 模型定义
│   ├── converters/
│   │   ├── convert_resnet50.py  #   ResNet50 → ONNX 流式转换
│   │   ├── convert_unet.py      #   UNet → ONNX 流式转换
│   │   └── export_c.py          #   ONNX → C 代码生成
│   ├── c_runtime/
│   │   ├── stream_runtime.h     #   流式推理 C 运行时头文件
│   │   └── stream_runtime.c     #   流式推理 C 运行时实现
│   ├── c_generated/             #   生成的 C 代码与内存规划
│   ├── tests/
│   │   ├── test_streaming.py    #   流式 vs 整段一致性
│   │   ├── test_streaming_c.py  #   C 运行时测试
│   │   ├── test_resnet50.py     #   ResNet50 测试
│   │   ├── verify_perlayer.py   #   逐层验证
│   │   └── verify_valid.py      #   端到端验证
│   └── analyze.py               #   模型分析工具
├── README.md
├── CLAUDE.md
└── .gitignore
```

## 竞争-互补耦合模块

### 核心思想

标准 1×1 卷积存在三个根本缺陷：**静态映射**、**线性瓶颈**、**无结构约束**。深层 CNN 的通道特征同时承载空间位置信息 (Where) 与语义表达信息 (What)，形成高度纠缠的"混合表征"。

我们将 C 个通道拆分为 L=8 个原型组，每组有 F=C/L 维特征，然后在组序列上应用标准多头注意力 (MHA)：

```python
from channel_attn.group_mha import GroupMHA

# channels 必须能被 8 整除
attn = GroupMHA(channels=256, L=8, num_heads=1, dropout_rate=0.1)
out = attn(x)  # x: (B, C, H, W) → (B, C, H, W)
```

**关键设计**：
- B×H×W 合并为 batch 维度，每个像素位置独立做组间注意力
- Softmax 在 L 维上实现组间竞争，Dropout 强制非独立互补
- 计算量约为标准 1×1 卷积的 50%，参数量约为 39%

## 流式推理

对于嵌入式场景，长序列需拆分为小块进行逐块推理。本项目提供完整方案确保流式输出与整段推理 **数学上完全等价**。

```python
# 模型训练与导出
python streaming/models/resnet50.py          # 定义模型
python streaming/converters/convert_resnet50.py  # 导出 ONNX
python streaming/converters/export_c.py      # 生成 C 代码 + 内存规划

# 测试验证
python streaming/tests/test_streaming.py     # 流式 vs 整段一致性
python streaming/tests/test_streaming_c.py   # C 运行时测试
python streaming/tests/verify_perlayer.py    # 逐层精度验证
```

C 运行时使用示例：

```c
#include "stream_runtime.h"
#include "resnetv2_stream.h"

// 初始化
StreamContext ctx;
stream_init(&ctx, &resnetv2_memory_plan);
stream_load_weights(&ctx, "resnetv2_weights.bin");

// 逐块推理
float input_chunk[CHUNK_SIZE];
float output_chunk[CHUNK_SIZE];
stream_infer(&ctx, input_chunk, output_chunk);

stream_free(&ctx);
```

## GSE — 粒化压缩-激励

通过两级组卷积空间下采样 (16×) 后进行通道挤压-激励，再上采样回原尺寸：

```python
from channel_attn.gse import GranularSE

se = GranularSE(channels=64, reduction=16)
out = se(x)  # 输入尺寸需为 16 的倍数
```

## 引用

详细理论推导见 [channel_attn/theory.md](channel_attn/theory.md)，流式推理方案见 [streaming/design.md](streaming/design.md)。

## 许可

MIT License
