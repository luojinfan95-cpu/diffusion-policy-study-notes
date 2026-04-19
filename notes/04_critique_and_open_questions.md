# 对 Diffusion Policy 的批判性思考与待解问题

> 精读的一个检验标准是能否指出论文的 limitation 和 open question。以下是我基于今天的理解能独立提出的几点。

---

## 1. 推理延迟的结构性限制

**问题**: 每次控制周期需要完整跑 K 次去噪迭代。训练 K=100，即使 DDIM 加速到 10 次，单次推理延迟仍 ~0.1s (Nvidia 3080)。

**影响场景**:
- **高频力控**（>100Hz）: 比如精密装配、接触丰富的双手操作，DP 无法实时响应
- **动态环境**: 物体高速运动时，延迟会让策略跟不上

**现有缓解**: DDIM、consistency models、noise schedule 优化。作者 Sec 9 明确承认这是 open limitation。

**我的观察**: 这是扩散模型的**本质代价**——用迭代去噪换表达能力。根本性改进可能需要单步扩散（如 consistency distillation）或改换到 flow matching 等新范式。

---

## 2. 行为克隆的继承缺陷

**问题**: DP 虽然用了强大的生成模型，但底层范式仍是 BC（behavior cloning）。所有 BC 的缺陷 DP 都有：

- **无自主探索**: 训练集没见过的状态无法优化
- **复合误差**: 推理时分布偏移累积
- **次优示教 → 次优策略**: 能力上限被数据质量限制
- **不能处理 OOD**: 真实场景中的新物体、新光照、新摩擦

**作者立场**: Sec 9 指出可以和 RL 结合（引用 Wang 2022, Hansen-Estruch 2023）。但论文本身没尝试。

**我的观察**: DP 的真正贡献是**证明扩散模型能作为 policy representation**，至于用 BC 还是 RL 训练是正交的。未来工作应该是 "Diffusion Policy + RL"，而不是 "纯 Diffusion BC"。这个方向目前已经有人在做。

---

## 3. 视觉编码器的未解之谜

**现象**: Sec 5.4 ablation 显示了一个反直觉结果：

| 策略 | Square (PH) Success Rate |
|---|---|
| 冻结 ImageNet ResNet-18 | 0.58 |
| 冻结 CLIP ViT-B/16 | 0.70 |
| **从头训练 ResNet-18** | **0.94** |
| 从头训练 ViT | 0.22 |
| 微调 CLIP ViT-B/16 | 0.98 |

**作者解释**: "Diffusion policy prefers different vision representation than what is offered in popular pretraining methods."——这是**描述，不是解释**。

**这是一个真正的 open question**:
- 为什么 DP 偏好不同的视觉表征？
- 什么叫 "DP-friendly" 的视觉表征？
- 能不能设计一个专门为动作预测服务的视觉预训练任务？

**我的猜想**: 扩散策略在每个 timestep 需要反复查询视觉特征，可能需要的是"**动作相关**"的表征（物体几何、接触点、affordance），而 ImageNet/CLIP 训的是"**语义相关**"的表征（物体类别、场景类型）。这两者不一样。如果是这样，专门为操作任务预训练的视觉编码器（类似 R3M 的思路但做对）可能是一个研究方向。

---

## 4. 条件注入的 heuristic 性质

**问题**: FiLM 注入每一层、用什么粒度的条件、条件压缩程度，全是超参数:

- 每层都注入 vs 只在特定层注入？
- 通道级 FiLM vs 像素级 FiLM？
- 观测直接进 FiLM vs 先经 MLP 压缩再进？
- CNN 的 FiLM vs Transformer 的 cross-attention——哪个更好取决于任务（Sec 3.1 承认这是 empirical 选择）

**对比**: Transformer 的 cross-attention 本身就是 learnable 的（attention weights 是从数据学出来的），更 adaptive。但 Transformer 又对超参敏感（attention dropout, weight decay 变化很大）。

**我的观察**: 这是深度学习普遍的问题，不是 DP 独有。但 DP 作为一个重要的 policy representation，理论上应该有更 principled 的条件注入方式——可能从"信息论"（互信息最大化）或"贝叶斯推断"（条件作为 prior）角度能给出指导。

---

## 5. 动作轨迹时序先验未显式建模

**问题**: 动作序列有很强的**时序平滑性先验**——相邻时间步的动作应该相似（除非有特殊事件如碰撞）。但 DP 的 1D CNN 只是**通用时序模型**，没有显式利用这个先验。

**表现**:
- U-Net 可以学到这个先验，但需要大量数据
- 数据少时，网络容易输出"锯齿状"动作（论文 Fig 3 的 BET 就是这个问题的极端版本）

**可能的改进方向**:
- 用 ODE/SDE 参数化动作轨迹：把动作序列表示为连续时间的随机过程
- Neural ODE 风格的轨迹参数化：保证输出自动平滑
- 多分辨率扩散：先去噪粗粒度轨迹骨架，再去噪细粒度细节

**我的观察**: 这个想法已经有人在做（continuous-time diffusion for trajectories），但还没在机器人操作上成熟。如果做成功，可能会让 DP 更 data-efficient。

---

## 6. 与我当前项目（天枢智驾）的潜在联系

我的冯如杯项目用的是 **VLM + 知识图谱 + GraphRAG** 处理自动驾驶的长尾场景。DP 和我的项目乍看无关，但有几个潜在的共通点:

**相似性**:
- 都是"从观测到动作"的端到端策略
- 都面临多模态问题（左转 vs 右转 vs 直行，同一路口都合理）
- 都需要时序一致性（不能一帧左转一帧右转）

**可能的融合方向**:
- **用扩散模型生成 ego 的未来轨迹序列**（替代 LightEMMA 的单步回归）—— DP 的核心优势正是"轨迹生成"
- **用知识图谱作为 DP 的条件**（类似 FiLM 但用图结构）—— 交通规则作为 soft constraint 注入去噪过程

这只是随手想想，没有可行性验证。但这说明精读一篇 "看起来不相关" 的论文有长远价值——**方法论的交叉往往在论文之间的缝隙里**。

---

## 7. 我没弄懂的点（需要补的）

诚实列出今天没完全吃透的地方:

- [ ] **ELBO 的 Jensen 不等式推导**: 今天只记了结论，没手推。概率论学完要回来补
- [ ] **DDIM 的具体推导**: 知道它加速推理，不知道数学上怎么做到"训练 100 步推理 10 步"。论文引用的 Song 2021 需要读
- [ ] **Noise schedule 的选择**: iDDPM 的 square cosine vs linear，为什么前者在机器人任务更好？Sec 3.3 只给了结论
- [ ] **Score matching 与 DDPM 的等价性**: Fig 1c 说 ε_θ ≈ ∇E(x)，论文里没推。需要读 Song & Ermon 2019
- [ ] **U-Net 为什么比普通 CNN 好**: 知道 skip connection 有用，但不知道信息论上是否有更深的解释
- [ ] **CNN vs Transformer 在 DP 上的精细差异**: 论文给了粗略建议（简单任务 CNN、高频任务 Transformer），但机理不清楚

---

## 8. 如果让我给这篇论文挑一个最重要的后续方向

**"Diffusion Policy 的视觉表征预训练"**

理由:
- 问题 3 是论文自己观察到的、没解释的现象，说明**确实有改进空间**
- 视觉表征是机器人学习的**通用基础设施**，做出来所有策略都受益
- 有明确的研究手柄：设计一个"为动作预测服务"的预训练目标
- 和 embodied AI 的大方向一致（具身表征学习）

相比之下，问题 1（推理加速）已经很多人在做，问题 2（BC→RL）技术路线清晰但苦工多，问题 5（时序先验）太底层不好落地。问题 3 是一个 "sweet spot" 研究方向。

这是我当下作为大一学生能想到的，可能很浅薄。真实研究者的判断必然更精细。
