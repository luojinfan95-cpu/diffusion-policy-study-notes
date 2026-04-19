# DDPM 数学推导

> 从前向过程定义到 L_simple 的完整推导，以及反向均值公式的来源。

---

## 0. 符号约定

| 符号 | 类型 | 含义 |
|---|---|---|
| $x_0$ | 随机向量 | 真实数据样本（图像 / 动作序列） |
| $x_t$ | 随机向量 | 第 t 步带噪样本 |
| $\varepsilon, \bar\varepsilon$ | 随机向量 | 标准高斯噪声 $\sim \mathcal{N}(0, I)$ |
| $\beta_t$ | 标量常数 | noise schedule，单步噪声强度 |
| $\alpha_t$ | 标量常数 | $1 - \beta_t$，单步保留比例 |
| $\bar\alpha_t$ | 标量常数 | $\prod_{s=1}^t \alpha_s$，累积保留比例 |
| $T$ | 整数 | 总扩散步数（典型值 1000） |

**关键区分**: $\alpha_t$ 是标量常数（人为设定的数字，如 0.9999），$x_t$ 是随机向量。两者不可混淆。

---

## 1. 前向过程

### 1.1 定义

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\; \sqrt{\alpha_t}\, x_{t-1},\; (1-\alpha_t) I)$$

等价的采样形式（用重参数化）：

$$x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\, \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0, I)$$

### 1.2 VP (Variance Preserving) 性质

假设 $x_0$ 方差为 1，则所有 $x_t$ 方差都保持为 1：

$$\text{Var}(x_t) = \alpha_t \cdot \text{Var}(x_{t-1}) + (1-\alpha_t) \cdot 1 = 1 \cdot 1 = 1$$

**这是选 $\sqrt{\alpha_t}$ 这个缩放的根本原因**——防止方差随 t 爆炸或塌缩。

### 1.3 极限行为

当 $t \to T$（且 T 足够大），由于均值被 $\sqrt{\alpha_t} < 1$ 每步压缩：

$$x_T \sim \mathcal{N}(0, I)$$

**这保证了推理起点是已知且可采样的分布**。

---

## 2. 前向闭式解

### 2.1 目标

训练时频繁需要"给定 $x_0$，采样任意 $t$ 步的 $x_t$"。按定义迭代 $t$ 次成本 O(t)，太慢。幸运的是有闭式解。

### 2.2 推导

从 $x_2$ 开始展开：

$$
\begin{aligned}
x_2 &= \sqrt{\alpha_2}\, x_1 + \sqrt{1-\alpha_2}\, \varepsilon_2 \\
&= \sqrt{\alpha_2}\,(\sqrt{\alpha_1}\, x_0 + \sqrt{1-\alpha_1}\, \varepsilon_1) + \sqrt{1-\alpha_2}\, \varepsilon_2 \\
&= \sqrt{\alpha_1\alpha_2}\, x_0 + \underbrace{\sqrt{\alpha_2(1-\alpha_1)}\, \varepsilon_1 + \sqrt{1-\alpha_2}\, \varepsilon_2}_{\text{两个独立高斯的和}}
\end{aligned}
$$

**用高斯叠加性质**（独立高斯 $\mathcal{N}(0, \sigma_1^2 I) + \mathcal{N}(0, \sigma_2^2 I) = \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2) I)$）合并噪声项，方差为：

$$\alpha_2(1-\alpha_1) + (1-\alpha_2) = 1 - \alpha_1\alpha_2$$

所以：

$$x_2 = \sqrt{\alpha_1\alpha_2}\, x_0 + \sqrt{1-\alpha_1\alpha_2}\, \bar\varepsilon_2,\quad \bar\varepsilon_2 \sim \mathcal{N}(0, I)$$

### 2.3 归纳结论

对任意 $t$，记 $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$：

$$\boxed{\; x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon,\quad \varepsilon \sim \mathcal{N}(0, I) \;}$$

等价的分布形式：

$$q(x_t | x_0) = \mathcal{N}(x_t;\; \sqrt{\bar\alpha_t}\, x_0,\; (1-\bar\alpha_t) I)$$

**工程价值**: 训练时一行代码实现 O(1) 加噪，不用走马尔可夫链。

---

## 3. 反向过程与贝叶斯反推

### 3.1 为什么要反推

推理时没有 $x_0$，只能走 $x_t \to x_{t-1}$ 的反向链。所以训练目标必须匹配反向条件分布 $q(x_{t-1} | x_t)$——**但这个分布没有直接定义**，我们只定义了前向。

**贝叶斯的作用**: 把"未知的反向"翻译成"已知的前向"。

### 3.2 贝叶斯公式（已知 $x_0$ 作为训练时条件）

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)}$$

三个分布我们都有：前向单步定义、前向闭式解、前向闭式解。代入并对 $x_{t-1}$ 配方后得到：

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1};\; \tilde\mu_t,\; \tilde\beta_t I)$$

其中真实反向均值为：

$$\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}(1-\alpha_t)}{1-\bar\alpha_t}\, x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x_t$$

### 3.3 用 $\varepsilon$ 重写均值

由前向闭式解反解 $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\varepsilon)/\sqrt{\bar\alpha_t}$，代入上式并化简：

$$\boxed{\; \tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon\right) \;}$$

**这就是为什么要用 ε-prediction**：它让反向均值有一个干净的 $(x_t, \varepsilon)$ 表达式。

### 3.4 模型参数化

让模型输出 $\varepsilon_\theta(x_t, t)$，反向均值参数化为完全相同的形式：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t, t)\right)$$

---

## 4. 从 ELBO 到 L_simple

### 4.1 目标：最大似然

想最大化 $\log p_\theta(x_0)$，但它涉及高维连续积分，intractable。

### 4.2 ELBO（可算的下界）

用 Jensen 不等式可以证明：

$$\log p_\theta(x_0) \geq \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

右边是 **Evidence Lower Bound**（ELBO），可以拆成 T 项求和。每项形如：

$$L_{t-1} = \mathbb{E}_q\left[D_\text{KL}\left(q(x_{t-1}|x_t, x_0) \,\|\, p_\theta(x_{t-1}|x_t)\right)\right]$$

**含义**: 让模型的反向过程匹配真实的反向过程。

### 4.3 两个同方差高斯的 KL

$$D_\text{KL}(\mathcal{N}(\mu_1, \sigma^2 I) \,\|\, \mathcal{N}(\mu_2, \sigma^2 I)) = \frac{1}{2\sigma^2}\|\mu_1 - \mu_2\|^2$$

**所以 KL 项本质就是均值的 MSE**。代入模型和真实反向均值：

$$L_{t-1} = \mathbb{E}\left[\frac{1}{2\sigma_t^2}\|\tilde\mu_t - \mu_\theta\|^2\right]$$

### 4.4 用 ε 重写 MSE

两个均值相减时 $x_t$ 项完全抵消，只剩 ε 的差：

$$\tilde\mu_t - \mu_\theta = \frac{1-\alpha_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}(\varepsilon - \varepsilon_\theta)$$

代回 $L_{t-1}$，合并所有系数：

$$L_{t-1} = \mathbb{E}\left[w_t \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]$$

其中 $w_t$ 是一个随 t 变化的权重。

### 4.5 L_simple

Ho 2020 经验发现：**丢掉 $w_t$，对所有 t 等权平均，效果反而更好**。得到：

$$\boxed{\; L_\text{simple} = \mathbb{E}_{x_0, t, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon,\; t)\|^2\right] \;}$$

**为什么丢权重能更好**: $w_t$ 在小 t 时大、大 t 时小，丢掉等价于相对提高大 t 的权重。而大 t 对应"粗糙去噪"阶段，决定生成样本的整体结构，对人眼感知质量更重要。ELBO 的"理论最优"和"生成质量最优"之间有 gap，L_simple 恰好更接近后者。

---

## 5. 训练与推理算法

### 5.1 训练（PyTorch 风格伪代码）

```python
def train_step(x_0, alpha_bar, model, optimizer):
    # 1. 采随机时间步和噪声
    t = torch.randint(0, T, (batch_size,))
    eps = torch.randn_like(x_0)
    
    # 2. 闭式解加噪
    x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * eps
    
    # 3. 网络预测 ε
    eps_pred = model(x_t, t)
    
    # 4. MSE loss
    loss = F.mse_loss(eps_pred, eps)
    
    # 5. 反向传播
    loss.backward()
    optimizer.step()
```

### 5.2 推理

```python
def sample(model, alpha, alpha_bar, sigma, T, shape):
    # 1. 从纯噪声起步
    x = torch.randn(shape)
    
    # 2. 反向去噪循环
    for t in reversed(range(1, T + 1)):
        eps_pred = model(x, t)
        mu = (1 / sqrt(alpha[t])) * (
            x - (1 - alpha[t]) / sqrt(1 - alpha_bar[t]) * eps_pred
        )
        z = torch.randn_like(x) if t > 1 else 0
        x = mu + sigma[t] * z
    
    return x  # x_0
```

---

## 6. 训练 vs 推理的已知/未知对照

| | 训练 | 推理 |
|---|---|---|
| $x_0$ | 已知（数据集） | 未知（要生成） |
| $t$ | 随机采样 | 从 T 递减到 1 |
| $\varepsilon$ | 自己采样且记录 | 网络预测 $\varepsilon_\theta$ |
| 中间 $x_{1..t-1}$ | 闭式解跳过 | 逐步生成 |
| 网络输出用途 | 进 loss | 进反向均值公式 |

这个对照表是理解 DDPM 设计的关键——**训练时有 $x_0$ 只是为了方便写 loss**，网络学完后推理时用 $\mu_\theta(x_t, t)$ 近似反向均值，不再需要 $x_0$。

---

## 7. 和 Diffusion Policy 的对应

DP 只是在 DDPM 基础上做了两个修改：

1. **x 换成动作序列**: $x_0 \to A_t^0$（形状 $T_p \times d$）
2. **加条件**: 所有分布和网络输入都加上观测 $O_t$

于是 DP 的训练 loss（论文 Eq 5）是：

$$L = \|\varepsilon - \varepsilon_\theta(O_t,\; A_t^0 + \varepsilon^k,\; k)\|^2$$

推理公式（论文 Eq 4）是：

$$A_t^{k-1} = \alpha(A_t^k - \gamma\varepsilon_\theta(O_t, A_t^k, k) + \mathcal{N}(0, \sigma^2 I))$$

其中 $\alpha, \gamma, \sigma$ 是论文把 DDPM 的复杂系数打包后的记号，和 DDPM 本质一致。
