# 论文概念到代码位置的映射

> 从论文公式和架构描述出发，定位到 `real-stanford/diffusion_policy` 仓库里的对应代码。

---

## 1. 训练 loss ↔ L_simple

**论文**: Eq (5), $L = \text{MSE}(\varepsilon^k, \varepsilon_\theta(O_t, A_t^0 + \varepsilon^k, k))$

**代码**: `diffusion_policy/policy/diffusion_unet_lowdim_policy.py::compute_loss`

**对应关系**:

| 论文概念 | 代码实现 |
|---|---|
| 采真实动作 $A_t^0$ | `nactions = batch['action']` |
| 采观测 $O_t$ | `nobs = batch['obs']` |
| 采随机 $k$ | `timesteps = torch.randint(0, num_train_timesteps, ...)` |
| 采噪声 $\varepsilon$ | `noise = torch.randn(nactions.shape)` |
| 闭式解加噪 | `noise_scheduler.add_noise(nactions, noise, timesteps)` |
| 网络预测 $\hat\varepsilon$ | `pred = self.model(noisy_trajectory, timesteps, global_cond=...)` |
| MSE loss | `F.mse_loss(pred, noise)` |

**注意**: `noise_scheduler.add_noise()` 是 HuggingFace `diffusers` 库里封装的闭式解，内部实现就是 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$。

---

## 2. 噪声预测网络 ↔ ConditionalUnet1D

**论文**: Fig 2b, CNN-based Diffusion Policy

**代码**: `diffusion_policy/model/diffusion/conditional_unet1d.py::ConditionalUnet1D`

**forward 接口**:

```python
def forward(
    self,
    sample: torch.Tensor,          # 带噪动作 A_t^k, 形状 (B, T_p, action_dim)
    timestep: Union[torch.Tensor, int],  # 时间步 k
    global_cond: torch.Tensor = None,    # 观测 O_t 特征（展平后）
) -> torch.Tensor:                 # 返回 ε̂, 形状同 sample
```

**内部关键组件**:

| 论文概念 | 代码位置 |
|---|---|
| 时间步 sinusoidal embedding | `SinusoidalPosEmb(dsed)` |
| 下采样 blocks | `self.down_modules` (ModuleList) |
| 上采样 blocks | `self.up_modules` (ModuleList) |
| 每个 block 的 residual 结构 | `ConditionalResidualBlock1D` |
| FiLM 调制 | `ConditionalResidualBlock1D::forward` 中的 `scale * x + bias` |

**FiLM 具体定位**:

```python
# ConditionalResidualBlock1D 内部
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, ...):
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels * 2),  # 输出 scale 和 bias 拼接
            ...
        )
    
    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)     # (B, 2*channels, 1)
        scale = embed[:, :channels, ...]    # 拆出 scale
        bias = embed[:, channels:, ...]     # 拆出 bias
        out = scale * out + bias            # ★ FiLM 调制 ★
        out = self.blocks[1](out)
        return out + self.residual_conv(x)
```

**对应论文 Fig 2b 的 `a·x + b`**: `scale * out + bias` 就是。

---

## 3. 推理循环 ↔ 反向去噪

**论文**: Eq (4), $A_t^{k-1} = \alpha(A_t^k - \gamma\varepsilon_\theta + \mathcal{N}(0, \sigma^2 I))$

**代码**: `diffusion_policy/policy/diffusion_unet_lowdim_policy.py::conditional_sample`

**对应关系**:

| 论文概念 | 代码实现 |
|---|---|
| 从纯噪声起步 $A_t^K$ | `trajectory = torch.randn(...)` |
| 设置去噪步数 K | `scheduler.set_timesteps(num_inference_steps)` |
| 反向循环 $k = K \to 1$ | `for t in scheduler.timesteps:` |
| 网络预测 $\varepsilon_\theta$ | `model_output = self.model(trajectory, t, ...)` |
| 反向均值公式 + 采样 | `trajectory = scheduler.step(model_output, t, trajectory).prev_sample` |

**重要**: 反向均值公式 $\mu_\theta = (x_t - \gamma\varepsilon_\theta)/\sqrt{\alpha_t}$ **封装在 `scheduler.step()` 里**。如果你用 `DDPMScheduler`，内部实现完整的 DDPM 反向；如果用 `DDIMScheduler`，用 DDIM 加速采样（训练 100 步，推理 10 步）。

---

## 4. Receding Horizon ↔ predict_action 外层

**论文**: Sec 2.3 "Closed-loop action-sequence prediction"

**代码**: `diffusion_unet_lowdim_policy.py::predict_action`

**核心逻辑**:

```python
def predict_action(self, obs_dict):
    # ... 处理观测 ...
    
    # 1. 调用 conditional_sample 生成完整 T_p 步动作
    nsample = self.conditional_sample(cond_data, cond_mask, ...)
    
    # 2. 从完整预测中切出前 T_a 步返回
    start = To - 1
    end = start + self.n_action_steps     # T_a 在 config 里叫 n_action_steps
    action = nsample[:, start:end]
    
    return {'action': action}
```

**外层调度**（环境 rollout 循环）在 `workspace/train_diffusion_unet_lowdim_workspace.py` 或 rollout utility 里：每次调用 `predict_action` 得到 $T_a$ 步动作，在环境里执行完这 $T_a$ 步后重新调用。

---

## 5. 配置关键参数

**文件**: `diffusion_policy/config/task/pusht_lowdim.yaml` 和 `train_diffusion_unet_lowdim_workspace.yaml`

| 配置项 | 含义 | PushT 值 |
|---|---|---|
| `horizon` | $T_p$（预测长度） | 16 |
| `n_obs_steps` | $T_o$（观测长度） | 2 |
| `n_action_steps` | $T_a$（执行长度） | 8 |
| `num_train_timesteps` | 训练去噪步数 K | 100 |
| `num_inference_timesteps` | 推理去噪步数 | 100 (lowdim) / 16 (image) |
| `noise_scheduler._target_` | DDPM 或 DDIM scheduler | `DDPMScheduler` |
| `model.diffusion_step_embed_dim` | 时间步 embedding 维度 | 128 |
| `model.down_dims` | U-Net 各层通道数 | [256, 512, 1024] |

---

## 6. 我的训练命令与环境

```bash
# 环境创建
mamba env create -f conda_environment.yaml
conda activate robodiff

# 训练命令（我实际用的）
python train.py \
    --config-dir=diffusion_policy/config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=pusht_lowdim

# 训练产出位置
data/outputs/pusht_lowdim_20260418_204730/
├── checkpoints/
│   ├── epoch=1050-test_mean_score=0.953.ckpt
│   ├── epoch=1500-test_mean_score=0.955.ckpt
│   ├── epoch=1750-test_mean_score=0.956.ckpt
│   ├── epoch=2250-test_mean_score=0.959.ckpt   ← best
│   ├── epoch=2550-test_mean_score=0.947.ckpt
│   └── latest.ckpt
├── logs.json.txt
└── .hydra/ (运行时配置)
```

**独立 eval 命令**:

```bash
python eval.py \
    --checkpoint data/outputs/pusht_lowdim_20260418_204730/checkpoints/epoch=2250-test_mean_score=0.959.ckpt \
    --output_dir data/eval_output_best
```

其实训练过程中 Hydra 已经每 50 epoch 自动做一次环境 rollout eval，top-k checkpoint 的 filename 里直接带 score，所以**独立 eval 通常不必要**，除非想改 rollout 数、用不同种子、或者可视化 rollout 视频。

---

## 7. 如何从代码反查回论文

倒着看代码时，一个有用的 heuristic:

| 代码 pattern | 可能对应论文的 |
|---|---|
| `add_noise`, `alphas_cumprod` | 前向闭式解 |
| `scheduler.step` | 反向均值公式 + 高斯采样 |
| `F.mse_loss(pred, noise)` | L_simple |
| `global_cond`, `cond_encoder` | FiLM / 条件注入 |
| `down_modules`, `up_modules`, `mid_modules` | U-Net 结构 |
| `n_action_steps`, `n_obs_steps`, `horizon` | 三个 horizon 参数 |
| `ConditionalUnet1D` vs `TransformerForDiffusion` | CNN 变体 vs Transformer 变体 |
