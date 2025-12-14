---
{"dg-publish":true,"permalink":"/科研/文献阅读/方法/表格推理/STaR/"}
---

>[!abstract] 论文概述 [STaR: Towards Cognitive Table Reasoning via Slow-Thinking Large Language Models](https://arxiv.org/abs/2511.11233)
>
>利用大语言模型进行表格推理仍然存在两个关键局限：（i）推理过程缺乏人类认知特有的深度和迭代精炼；（ii）推理过程表现出不稳定性。
>
>文章提出了 STaR（表格推理慢思考），一种实现认知表格推理的新框架，LLM 通过显式建模逐步思考和推理，赋予了慢思考能力。
>
>在训练过程中，STaR 采用两阶段的难度感知强化学习（DRL），在复合奖励下从简单到复杂查询逐步学习。在推断过程中，STaR 通过整合 token 级执行度和答案一致性，进行轨迹级不确定性量化，从而选择更具可信度的推理轨迹。
>
>仓库链接 [zhjai/STaR](https://github.com/zhjai/STaR)

## Intriduction

有效解决诸如表格问答和事实验证等基础任务仍然具有挑战性，需要整合多种能力：精确的信息检索、自然语言理解、多步逻辑推理和准确的数值计算。

为了引导大语言模型生成连贯且可解释的推理轨迹，文章提出了 STaR（表格推理慢思考）框架。在训练过程中，STaR 实施了一个精心设计的范式，结合了慢思考的数据集构建与两阶段的难度感知强化学习（DRL）。

首先通过自我验证的高质量演示建立基础推理模式，然后逐步挑战模型，从处理简单查询到掌握需要在整个表格结构中综合信息的复杂多步推理。该方法采用动态样本过滤和复合奖励，有效引导学习过程。在推断过程中，STaR 不依赖单次生成，而是生成多条推理轨迹，并采用复杂的不确定性量化（UQ）。通过将 token 级置信度与答案一致性融合，该框架识别并选择了最可信的推理路径，有效地将潜在的 pass@k 潜力转化为可靠的 pass@1 表现。

## Methodology

STaR 框架包含三部分：慢思考 SFT 数据、两阶段难度感知强化学习，以及用于可靠路径选择的不确定性量化。

### Framework Overview

![[attenchments/Pasted image 20251210235411.png#pic_center]]

STaR 是一个认知表格推理框架，集成了三个核心组成部分：缓慢思考数据集构建、两阶段难度感知强化学习和轨迹级不确定性量化。如图 1 所示，该框架采用结构化提示和自我验证机制构建高质量数据集。STaR 随后采用两阶段强化学习范式：第一阶段从简单示例基础训练，采用最小步数；第二阶段通过动态查询过滤器的迭代训练逐步掌握困难样本。在推理过程中，STaR 生成多条推理轨迹，并通过讲 token 层面熵和答案一致性整合来量化其可靠性，从而通过加权融合选择最可靠的路径。这种轨迹级不确定性量化确保了表格推理任务的准确性和稳定性。

### Slow-Thinking Dataset Construction

通过答案感知生成方法，构建了 WikiTableQuestions、HiTab 和 FinQA 的高质量训练数据集。具体来说，为 DeepSeek-R1 提供了表格、问题和正确答案，促使其生成完整的推理轨迹。

文章数据构建的一个关键贡献是自我验证机制，模型将生成的答案与真实数据进行比较，并自动过滤输出不一致的样本。这种方法不仅确保推理轨迹与最终答案的对齐，还能从训练集中提出可能存在歧义或标注错误的数据，显著提升数据质量和可靠性。~~【原来这个也能算是贡献吗】~~

生成的演示遵循结构化格式：`<think>reasoning process</think><answer>final answer</answer>`，其中思考部分包含详细的思路推理，答案部分提供结构化的 JSON 输出。

### Reinforcement Learning

#### Enhanced GRPO

采用 DAPO 增强版 GRPO 框架，去除 KL 散度惩罚并采用非对称裁剪界限。去除 KL 散度使模型在发现对表格推理至关重要的复杂推理模式时能够偏离初始分布，而非对称裁剪策略则鼓励探索对新颖推理策略至关重要的低概率 token。这些修改对于表格推理尤为重要，因为不同表格结构中多样的问题解决方法需要灵活的策略调整。最终 GRPO 目标为：

$$
\begin{aligned}

\mathcal{J}_{\mathrm{GRPO}}(\theta) & =\mathbb{E}_{(q, a) \sim \mathcal{D},\left\{o_{i}\right\}_{i=1}^{G} \sim \pi_{\theta_{\text {old }}}(\cdot \mid q)} \\

& {\left[\frac { 1 } { \sum _ { i = 1 } ^ { G } | o _ { i } | } \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { | o _ { i } | } \operatorname { m i n } \left(r_{i, t}(\theta) \hat{A}_{i, t}\right.\right.} \\

& \left.\left.\operatorname{clip}\left(r_{i, t}(\theta), 1-\varepsilon_{\text {low }}, 1+\varepsilon_{\text {high }}\right) \hat{A}_{i, t}\right)\right],

\end{aligned}
$$

其中 $\epsilon_{high}>\epsilon_{low}$（例如 $\epsilon_{high}=0.28~~\epsilon_{low}=0.2$），使得更积极的更新以实现有利的推理模式。 

#### Difficulty-Aware Training

![[attenchments/Pasted image 20251211095957.png#pic_center | 400]]

训练策略采用两阶段的 DRL，通过将简单推理任务与复杂任务分离（如图 2 所示）从而提高学习效率。根据 SFT 模型计算的 $pass@k_1=0.6$ 阈值对数据集进行拆分，创造了简单（~10000 样本）和困难（~10000 样本）训练子集。

基础训练阶段（第一阶段）关注易于理解的数据集，学习率高（例如 $1\times10^{-5}$），能够在最小的训练步骤内快速实现约 80% 的性能。值得注意的是，采用单阶段方法需要模型大量时间学习和自适应过滤这些简单样本，且学习率较低，从而降低训练效率。

渐进训练阶段（第二阶段）专注于学习率较低（例如 $1\times10^{-6}$）的困难数据集，并基于实时的 $pass@k_2$ 评估进行动态样本过滤。自适应路由机制的工作原理如下：为避免过拟合，$pass@k_2=1.0$ 的样本被排除，$pass@k_2<1.0$ 的样本被放入评审池进行定期重新评估，只有 $pass@k_2<0.8$ 的样本才能获得有效的 GRPO 更新。这种动态过滤让计算资源专注于真正困难的推理问题：

$$
\mathcal{J}_{\mathrm{GRPO}} = \mathbb{E}_{s \in \mathcal{S}_{\text{active}}} \left[ \mathcal{L}_{\mathrm{GRPO}}(s) \right] \; \text{s.t.} \; 0 < \text{pass}@k_{2}(s) < 0.8
$$

其中 $\mathcal{S}_{\text{active}}$ 表示主动训练的样本子集。该策略确保训练资源集中于模型当前适当难度水平的样本，避免过于简单的样本浪费计算，同时保持先前获得的能力。

#### Reward Function Design

采用复合奖励函数，通过三个加权组成部分评估结构合规性和内容准确性。

- 格式合规性（0.2）确保 `<think>` 和 `<answer>` 标签正确对齐，且答案部分为有效的 JSON 格式。这一部分至关重要，因为错误的输出根本无法解析或评估。
- 部分正确率（0.3）在预测与多重答案的真实性列表中的任何部分相符时得分。这通过在训练中提供渐进奖励，鼓励逐步达到完全准确的水平。
- 完全正确度（0.5）为与正确答案的精确匹配提供了最强的学习信号。

最终归一化的奖励在结构要求和准确性之间 $R = 0.2 \times R_{\text{format}} + 0.3 \times R_{\text{partial}} + 0.5 \times R_{\text{complete}}$ 取得平衡。它确保模型能够学习生成可解析且正确的输出，同时通过部分学分机制促进渐进式学习。

### Uncertainty Quantification

#### Token-Level Confidence Metrics

通过计算模型内部概率分布中的 token 级置信度量化个体推理轨迹的可靠性。对于每个由 token 组成 $\{t_1, t_2, ..., t_n\}$ 的生成轨迹，计算平均对数概率和平均熵：

$$
\text{logprob}(y) = \frac{1}{n} \sum_{i=1}^{n} \log p(t_i | t_{<i}, x)
$$

$$
\text{entropy}(y) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{v \in \mathcal{V}} p(v | t_{<i}, x) \log p(v | t_{<i}, x)
$$

其中 $x$ 表示输入查询，$\mathcal{V}$ 表示词汇表，$p(t_i | t_{<i}, x)$ 是给定前 $i$ 个词生成 token $t_i$  的概率。

#### Consistency-Confidence Fusion Algorithm

![[attenchments/Pasted image 20251211103449.png#pic_center | 400]]

文章的轨迹级不确定性量化整合了 token 级执行度和答案级一致性，解决了单独使用任一指标的局限性。

仅依赖答案一致性（多数票）会忽视一些罕见但正确的推理路径。这在一些复杂案例中尤为明显，大多数路径通向合理但错误的解决方案。相反，仅关注 token 层级置信度可能导致模型对其错误推理过于自信的偏差。因此采用加权融合策略，平衡两种信号以实现稳健的轨迹选择，如算法 1 所示。

融合策略的数学表述结合了三个归一化组件：

$$
S(a) = 0.25 \cdot \frac{|G_a|}{|G_{\text{max}}|} + 0.2 \cdot \frac{\bar{C}_a}{C_{\text{max}}} + 0.55 \cdot \frac{C_a^{\text{max}}}{C_{\text{max}}^{\text{global}}}
$$

其中 $|G_a|$ 表示答案的一致性计数 $a$，$\bar{C}_a$ 是组 $a$ 内的平均置信度，$C_{\text{max}}$ 表示组 $a$ 内的最大置信度得分。

## Experiments

**参数设置**

SFT 阶段采用 batch size=256, lr=$1\times10^{-5}$, epoch=3

利用 Qwen3 0.6B 模型惊醒两阶段 GRPO 训练的数据集拆分，$pass@32$ 精度阈值设置为 0.6。
第一阶段通过 batch size=512 快速建立基础能力，每个样本采样 5 次，lr=$1\times10^{-5}$。
第二阶段 batch size=256，采样 8 次，lr=$1\times10^{-6}$ with decay rate=0.01。
两阶段温度均为 1.0，生成长度 4096 token，非对称裁剪界限 \[0.2, 0.28\]

推理过程中每个查询生成 8 个采样，温度为 0.6， 最大长度 4096 token。

**结果**

*主实验*

![[attenchments/Pasted image 20251211114717.png#pic_center]]

表 1 中是所有的评估结果，结果显式 STaR 在所有基准测试中均达到了最优性能。同时可以看到 STaR 的泛化能力表现出色。

*训练组件分析*

![[attenchments/Pasted image 20251211115118.png#pic_center | 400]]

表 2 是 Qwen3-0.6B 在不同基准测试下不同训练配置的性能对比。结果显示出互补优势：仅用 SFT 训练在数据集上提供稳定的基线性能，但缺乏复杂查询的推理深度；而仅用强化学习训练在特殊和特定数据集（如 WTQ）上表现优异，但存在不一致性，尤其是在 TabFact 上，性能大幅下降。完整的 SFT+RL 流水线在所有基准测试中都实现了卓越的性能。

![[attenchments/Pasted image 20251211115403.png#pic_center | 400]]

图 3 是训练的动态结果，显示的是不同训练步数下的模型的性能。可以看到 Qwen3-0.6B 和 Qwen3-8B 模型在 WTQ 和 HiTab 数据集的 140 个训练步骤中表现持续提升。前 20 步（第一阶段）表现出快速的性能提升，模型能迅速从简单样本中学习，而随后的 120 步（第二阶段）则随着模型处理越来越复杂的推理模式，逐步但稳步地提升。

![[attenchments/Pasted image 20251211115533.png#pic_center | 400]]

图 4 是在强化学习中使用单阶段于两阶段训练曲线的对比【论文没说具体比的是哪个模型】。可以看到，两阶段方法展现了明显优势：在第 20 步，第一阶段的快速学习使我们的模型能够超越第一阶段基线，而基线进展更为缓慢。在 WTQ 中，单阶段方法在第 120 步左右趋于稳定，而两阶段方法持续改进，显示出更强的长期优化潜力。

![[attenchments/Pasted image 20251211115746.png#pic_center | 400]]

表 3 是两阶段强化学习的消融实验结果。可以看到移除第二阶段会导致所有数据集的性能大幅下降，HiTab（-14.34%）和 FinQA（-8.97%）的下降尤为显著，表明对硬样本进行聚焦训练对于实现强劲最终性能至关重要。

![[attenchments/Pasted image 20251211115929.png#pic_center | 400]]

表 4 是不确定性量化对实验结果的影响。













