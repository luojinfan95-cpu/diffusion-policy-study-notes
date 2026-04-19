# Diffusion Policy 架构拆解

> 五个组件、训练与推理两条数据流、CNN/Transformer 两个变体的功能对应。

---

## 1. 五个组件

| # | 组件 | 输入 | 输出 | 何时存在 |
|---|---|---|---|---|
| 1 | 观测编码器 | 原始观测（image / lowdim） | 观测特征 $O_t$ | 训练 + 推理 |
| 2 | 时间步 embedding | 整数 $k \in \{0, ..., K-1\}$ | sinusoidal 向量 | 训练 + 推理 |
| 3 | 动作序列 | — | 带噪动作 $A_t^k \in \mathbb{R}^{T_p \times d}$ | 训练时从数据 + 噪声合成；推理时迭代生成 |
| 4 | 噪声预测网络 $\varepsilon_\theta$ | $(A_t^k, O_t, k)$ | $\hat\varepsilon$，同形状 | 训练 + 推理 |
| 5 | Receding Horizon Controller | $A_t^0$（去噪完的整段） | 执行前 $T_a$ 步 | **仅推理** |

**关键观察**: 组件 5 只在推理时存在。训练时不需要外层调度——训练只负责让 ε_θ 学会预测噪声。

---

## 2. 三个 Horizon 参数

Fig 2a 的时间轴设计：

| 参数 | 含义 | PushT 典型值 | 作用 |
|---|---|---|---|
| $T_o$ | 观测历史长度 | 2 | 网络每次看最近几步观测（允许差分出速度等状态信息） |
| $T_p$ | 动作预测长度 | 16 | 一次生成多长的动作序列（**保证时序一致性**） |
| $T_a$ | 动作执行长度 | 8 | 每次重规划前实际执行多少步（**保证响应性**） |

**Receding horizon 的核心权衡**: $T_p$ 要大（网络能看到整段轨迹，做出 mode-consistent 的承诺），$T_a$ 要小（快速响应环境变化）。论文 Fig 5 显示 $T_a = 8$ 是大多数任务的甜点。

---

## 3. 训练数据流

```
┌─── 从数据集采一条 demo 片段 ───┐
│                                 │
│  观测片段 (T_o 步)              │
│         │                       │
│         ▼                       │
│    [观测编码器]                 │
│         │                       │
│         ▼                       │
│  观测特征 O_t                   │
│                                 │
│  真实动作片段 A_t^0 (T_p 步)    │
└─────────┬───────────────────────┘
          │
     随机 k ∼ Uniform{0,...,K-1}
     随机 ε ∼ N(0, I)
          │
          ▼ 闭式解
     A_t^k = √ᾱ_k·A_t^0 + √(1-ᾱ_k)·ε
          │
          ▼
 ┌────────────────────────────────┐
 │ ε_θ(A_t^k, O_t, k) → ε̂         │  ← 噪声预测网络
 │ (CNN U-Net + FiLM 或 Transformer)│
 └────────────────┬───────────────┘
                  │
                  ▼
          Loss = ||ε - ε̂||²
                  │
                  ▼
          反向传播更新 θ
```

**三个要点**:
- 加噪加在**真实动作 $A_t^0$** 上，不是观测上
- 观测 $O_t$ 是**条件**（干净的、不加噪），通过 FiLM / cross-attention 注入
- 网络接收三个输入: 带噪动作、观测特征、时间步

---

## 4. 推理数据流

```
时刻 t: 从环境拿到观测 O_t
         │
         ▼
    [观测编码器]
         │
         ▼
      O_t 特征                    ← 只算一次 (关键优化)
         │
         ▼
    A_t^K ∼ N(0, I)               ← 从纯噪声起步
         │
┌────────┴─── for k = K-1, ..., 0 ────────┐
│                                          │
│    ε̂ = ε_θ(A_t^k, O_t, k)                │  ← 每次迭代重用同一个 O_t
│         │                                 │
│         ▼                                 │
│    μ_θ = (x_t - γ·ε̂)/√α                  │
│         │                                 │
│         ▼                                 │
│    A_t^{k-1} = μ_θ + σ·z, z∼N(0,I)        │  (k=0 时 z=0)
│                                          │
└────────────────┬─────────────────────────┘
                 │
                 ▼
         A_t^0 (去噪完的完整 T_p 步)
                 │
                 ▼
       [Receding Horizon Controller]
                 │
                 ▼
    执行前 T_a 步: a_t, a_{t+1}, ..., a_{t+T_a-1}
                 │
                 ▼
    T_a 步后 → 时刻 t+T_a，重新观测，重新推理
```

**三个要点**:
- 观测编码器只跑一次（论文 Sec 2.3 明确指出这是 visual conditioning 的主要优化）
- 去噪循环 K 次，每次都用**同一个** $O_t$
- Receding horizon 是最外层调度，不在去噪循环内

---

## 5. CNN 变体内部（1D U-Net + FiLM）

### 5.1 整体结构

```
输入: A_t^k 形状 (T_p=16, action_dim=2)            channels=2
  ▼ Conv1D 升通道
(16, 256)                                          channels=256
  ▼ 下采样 Block 1
(8, 256)
  ▼ 下采样 Block 2
(4, 512)
  ▼ 下采样 Block 3
(2, 1024)
  ▼ bottleneck
(2, 1024)
  ▼ 上采样 Block 3' (+ skip from Block 3)
(4, 512)
  ▼ 上采样 Block 2' (+ skip from Block 2)
(8, 256)
  ▼ 上采样 Block 1' (+ skip from Block 1)
(16, 256)
  ▼ Conv1D 降通道
输出: ε̂ 形状 (16, 2)                              channels=2
```

### 5.2 ConditionalResidualBlock1D 内部

```
x ──► Conv1D ──► GroupNorm ──► FiLM ──► Mish ──► (Residual) ──► 输出
                                 ▲
                                 │
                    [O_t, embed(k)] ──► Linear ──► (scale, bias)
```

### 5.3 FiLM 机制详解

**公式**: $\text{output} = \text{scale} \cdot x + \text{bias}$

**参数生成**: 一个 Linear 层 `[O_t; embed(k)] → concat → Linear → (scale, bias)`。每层 U-Net block 有自己独立的 Linear 参数。

**调制粒度**: 通道级。假设 scale, bias 形状都是 (channels,)，主干特征 x 形状 (T_p, channels)：
```python
# 对每个通道 c：
x[:, c] = scale[c] * x[:, c] + bias[c]
# 同一通道的所有 T_p 个时间步共享同一组 (scale, bias)
```

**直觉**: scale 控制"哪些通道被放大"，bias 控制"基准激活水平"。不同观测 → 不同 (scale, bias) → 激活主干网络中不同的动作模式。**这就是观测影响动作的机制**。

**如果去掉 FiLM**: ε_θ 退化为只依赖 $A_t^k$ 和 $k$，变成无条件动作生成器——会生成"某种合理动作"但完全不看当前场景。DP 就不是 policy 了。

---

## 6. Transformer 变体（minGPT + cross-attention）

### 6.1 结构对应

| CNN 变体 | Transformer 变体 |
|---|---|
| 主干 = 1D U-Net (Conv1D + skip connections) | 主干 = causal self-attention |
| 条件注入 = FiLM (scale·x + bias) | 条件注入 = cross-attention |
| 时序耦合 = 卷积核的局部感受野，层叠后扩大 | 时序耦合 = self-attention 一层全局 |

### 6.2 输入形式

```
Tokens:  [embed(k), A_t^k[0], A_t^k[1], ..., A_t^k[T_p-1]]
          ↑ 时间步     ↑ 动作 tokens（被 causal mask 约束）

观测 O_t 被 shared MLP 编码后作为 memory，通过每个 decoder block 的 cross-attention 被查询
```

### 6.3 权衡

- CNN: 训练稳定、超参鲁棒、出的开箱即用
- Transformer: 容量大、表达能力强、适合高频动作变化的场景，但**对超参（attention dropout, weight decay）极敏感**

论文 Sec 3.1 推荐：**先用 CNN 试水，performance 不够再换 Transformer**。

---

## 7. 视觉编码器（仅 image-based 任务）

标准 ResNet-18 + 三处修改（Sec 3.2）:

1. **全局平均池化 → spatial softmax pooling**（保留空间信息，来自 Mandlekar 2021）
2. **BatchNorm → GroupNorm**（与 EMA 搭配训练稳定）
3. **不预训练，端到端训练**（论文 Sec 5.4 ablation 显示从头训比冻结 CLIP 更好）

每个相机视角用独立编码器，不同时间步的图像独立编码再 concat 成 $O_t$。

---

## 8. 关键形状速查表（PushT lowdim）

| 量 | 形状 | 含义 |
|---|---|---|
| 观测 $O_t$ | (T_o=2, obs_dim=5) | 最近 2 步状态 |
| 动作序列 $A_t$ | (T_p=16, action_dim=2) | 未来 16 步动作 |
| 时间步 $k$ | scalar | 0..99 |
| $\varepsilon$, $\hat\varepsilon$, $A_t^k$ | (16, 2) | 都同形状，这是 U-Net 设计的强制要求 |
| 执行的动作 | (T_a=8, 2) | 前 8 步，送给环境 |

**核心约束**: ε_θ 的输入输出同形状——这是 U-Net 架构的自然产出，也是"预测噪声"任务的内在要求。
