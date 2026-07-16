# Flow CNN Channel Attention

基于通道组竞争-互补耦合的 CNN 注意力机制与流式推理框架。

## 概览

本项目包含三个核心部分：

1. **竞争-互补耦合模块** — 替代标准 1×1 卷积的通道分组多头注意力机制
2. **流式推理框架** — 确保逐块流式推理与整段推理数值严格一致的 ONNX 转换与 C 运行时
3. **粒化 SE 模块 (GSE)** — 基于空间下采样的粒化压缩-激励注意力

## 项目结构

```
├── 竞争互补.py              # 竞争-互补耦合模块 (GroupMHA)
├── 竞争互补.md              # 理论推导与设计文档
├── 方案.md                  # 流式推理数值一致性方案
├── GSE/
│   └── gse.py               # 粒化压缩-激励模块 (GranularSE)
├── resnet50_model.py        # ResNet50 模型定义
├── unet_model.py            # UNet 模型定义
├── convert_resnet50.py      # ResNet50 → ONNX 转换
├── convert_unet.py          # UNet → ONNX 转换
├── export_streaming_c.py    # ONNX → C 代码生成
├── c_runtime/
│   ├── stream_runtime.h     # 流式推理 C 运行时头文件
│   └── stream_runtime.c     # 流式推理 C 运行时实现
├── c_generated/
│   ├── resnetv2_stream.c/h  # ResNetV2 生成的流式 C 代码
│   └── unet_stream.c/h      # UNet 生成的流式 C 代码
├── test_streaming.py        # 流式推理一致性测试
├── test_streaming_c.py      # C 运行时测试
├── _verify_perlayer.py      # 逐层验证脚本
├── verify_valid.py          # 端到端验证脚本
├── analyze.py               # 模型分析工具
└── test_resnet50.py         # ResNet50 测试
```

## 竞争-互补耦合模块

### 核心思想

标准 1×1 卷积存在三个根本缺陷：**静态映射**、**线性瓶颈**、**无结构约束**。深层 CNN 的通道特征同时承载空间位置信息 (Where) 与语义表达信息 (What)，形成高度纠缠的"混合表征"。

我们将 C 个通道拆分为 L=8 个原型组，每组有 F=C/L 维特征，然后在组序列上应用标准多头注意力 (MHA)：

```python
from 竞争互补 import GroupMHA

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
python resnet50_model.py          # 定义模型
python convert_resnet50.py        # 导出 ONNX
python export_streaming_c.py      # 生成 C 代码 + 内存规划

# 测试验证
python test_streaming.py          # 流式 vs 整段一致性
python test_streaming_c.py        # C 运行时测试
python _verify_perlayer.py        # 逐层精度验证
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
from GSE.gse import GranularSE

se = GranularSE(channels=64, reduction=16)
out = se(x)  # 输入尺寸需为 16 的倍数
```

## 引用

详细理论推导见 [竞争互补.md](竞争互补.md)，流式推理方案见 [方案.md](方案.md)。

## 许可

MIT License
