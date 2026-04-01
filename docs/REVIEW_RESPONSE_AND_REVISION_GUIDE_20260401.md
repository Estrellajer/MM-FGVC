# 审稿问题总结与修稿回复草案（2026-04-01）

本文档用于把当前审稿意见、已完成实验、以及建议的论文修改动作整理成一份可直接用于修稿的材料。

说明：你给出的“原始 10 个问题”实际包含 `11` 条审稿点，下面按 `11` 条完整整理。

## 一、这次可用的实验与证据

### 1. 新增的简单 FGVC frozen-feature 基线

本次新增了两个 reviewer-targeted 的简单基线，并已经接入现有实验框架：

- `Whitened NCM`
- `Ridge Probe`

实现位置：

- `src/methods/frozen_feature.py`
- `conf/method/whitened_ncm.yaml`
- `conf/method/ridge_probe.yaml`

结果记录：

- `swap/paper/records/fgvc_review_simple_20260401_151500.md`
- `swap/paper/records/fgvc_review_simple_comparison_20260401_151500.md`

对比结果如下：

| Task | Zero-shot | SAV | Whitened NCM | Ridge Probe | RSEv2 |
| --- | --- | --- | --- | --- | --- |
| eurosat_fgvc | 0.6000 | 0.7100 | 0.6900 | 0.7100 | 0.7600 |
| pets_fgvc | 0.9500 | 0.9700 | 0.9700 | 0.9700 | 0.9600 |
| cub_fgvc | 0.9100 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| mean | 0.8200 | 0.8933 | 0.8867 | 0.8933 | 0.9067 |

这组实验最重要的结论是：

- 新增的简单 frozen-feature 基线是有竞争力的。
- 但它们并不能完全解释 HiRe / RSEv2 的收益。
- 在三任务均值上，`RSEv2 = 0.9067`，高于 `Ridge Probe = 0.8933` 和 `Whitened NCM = 0.8867`。
- 区分最明显的是 `eurosat_fgvc`，这里更能体现多层级读取带来的增益。

### 2. 已有的 RSEv2 消融实验

结果记录：

- `swap/paper/records/c2_ablations_20260326_144544.md`

与这次修稿最相关的几项如下：

| Variant | Mean Primary |
| --- | --- |
| RSEv2 | 0.7225 |
| RSEv2-HeadOnly | 0.6917 |
| RSEv2-EqualWeight | 0.7108 |
| RSEv2-Top1 | 0.7000 |
| RSEv2-Top4 | 0.7025 |
| RSEv2-Top8 | 0.7058 |
| RSEv2-All | 0.7225 |

这组实验支持两个结论：

- 多层级表示聚合确实有帮助，不能简单退化为单一 head 或单一层。
- `margin` 加权是经验上有效的，但默认配置并没有形成“强稀疏”的自动筛选机制；`All` 往往已经最好。

### 3. 当前主文中已经存在、可直接利用的内容

主文 `main.tex` 中已经有一些对审稿意见有帮助的材料：

- 已经开始对 `DPI` 做 disclaimer。
- 已经写入 BLINK 的失败边界和 `closed-set discrimination` 的范围判断。
- 已经在 limitations 中承认 read-based framework 不替代 generative reasoning。

所以现在不是“从零开始修”，而是“把已有内容再收紧、再对齐、再补上关键证据”。

## 二、逐条问题判断与建议回复

下面每一条都包含四部分：

- `问题判断`：这条到底是真缺陷，还是主要是表述问题。
- `可用证据`：现在有哪些实验或文本可以支撑回应。
- `建议回复`：可直接改写成 rebuttal / cover letter / 修稿说明。
- `建议改稿动作`：具体应该在论文里改什么。

---

## 1. 理论主张被夸大（DPI / 信息论论证）

### 问题判断

这条 `部分成立`，而且主要是 `理论口径过强`，不是定理本身错误。

审稿人的核心点是对的：  
`I(X; z) >= I(X; Y_gen)` 只能说明中间表示保留了不少于生成输出的输入信息，不能直接推出它对分类标签 `Y` 更有用。  
也就是说，`Proposition 3.1` 最多只能作为方法动机，不能作为“分类更优”的严格证明。

### 可用证据

- 主文已经开始加入 disclaimer。
- 当前方法有效性的主要证据实际上来自实验，而不是 DPI 推导本身。

### 建议回复

我们同意审稿人的指出。当前的 DPI 论证只能说明中间表示至少保留了不低于生成输出的输入信息，而不能直接证明其对分类标签更具可用性。我们将在修订版中明确将该命题降格为方法动机，而不再将其表述为对分类优势的理论证明。HiRe 在 closed-set classification 上的有效性将主要由经验结果支持。

### 建议改稿动作

- 把 `Proposition 3.1` 从“证明性命题”改成 `motivation / remark`。
- 把文中类似 “preserve more task-relevant information” 的说法改成：
  - `more linearly accessible discriminative signal for closed-set classification`
  - 或 `retains discriminative cues that can be read out more effectively for closed-set label matching`
- 删除任何从 DPI 直接推出“分类更优”的表述。

### 可直接落文的英文改写方向

Instead of interpreting Proposition 3.1 as a proof of label-level superiority, we use it only as a motivation: internal states retain at least as much information about the input as the decoded output, which suggests that they may expose discriminative cues that are easier to read out for closed-set classification. The actual usefulness for classification is established empirically rather than by the DPI alone.

---

## 2. 马氏距离最优性的高斯假设存疑

### 问题判断

这条 `部分成立`，本质上也是 `理论口径过强`。

`P(z|y=c)=N(\mu_c,\Sigma)` 下的 Bayes-optimal 结论在条件成立时没有问题，但不能写成对真实 Transformer 激活分布的现实最优性证明。审稿人的质疑针对的是“从条件性结论跳到了现实最优”。

### 可用证据

- 现有 `Proposition 3.2` 作为条件性命题是可以保留的。
- 不需要额外实验来证明“激活真的是高斯”，最稳的处理方式是降调。

### 建议回复

我们同意审稿人的担忧。命题 3.2 的目的只是说明，在 shared-covariance Gaussian approximation 下，Mahalanobis/LDA 具有一个清晰的判别学解释。我们并不主张真实 LMM 激活严格满足该分布假设，因此修订版将把该结论明确表述为 metric 设计动机，而非对实际特征空间最优性的严格证明。

### 建议改稿动作

- 将 “optimal linear separation” 改为：
  - `a principled metric choice under a shared-covariance Gaussian approximation`
- 删除任何暗示“真实 Transformer 表征因此最优”的表述。
- 可以补一句：
  - `We use this result as a modeling approximation rather than a literal distributional claim about LMM activations.`

---

## 3. 缺乏与更强表示学习基线的比较

### 问题判断

这条 `成立，但现在已经被部分补上`。

审稿人明确提了三类替代方法：

- 线性探测 / 岭回归
- 白化最近类均值
- 轻量级学习融合

你现在已经补上了前两类里最容易、也最对 reviewer 胃口的版本：`Ridge Probe` 和 `Whitened NCM`。这一点非常关键。

### 可用证据

新增实验：

- `swap/paper/records/fgvc_review_simple_comparison_20260401_151500.md`

关键结果：

- `RSEv2 = 0.9067`
- `Ridge Probe = 0.8933`
- `Whitened NCM = 0.8867`

尤其是：

- `eurosat_fgvc`: `RSEv2 0.7600 > Ridge Probe 0.7100 > Whitened NCM 0.6900`

这说明：

- 改进不能完全被“单层 frozen-feature + 更好读出器”解释。
- 多层级读取依然提供了额外增益。

### 建议回复

我们感谢审稿人的建议，并在修订版中新增了两个简单但强有力的 frozen-feature 基线：Whitened NCM 和 closed-form Ridge Probe。二者都使用与 HiRe 相同的 support/query protocol，并直接作用于冻结的 LMM 表征。在 `qwen2_vl` 的 `eurosat_fgvc / pets_fgvc / cub_fgvc` 三个任务上，HiRe/RSEv2 取得 `0.9067` 的平均准确率，高于 `Ridge Probe (0.8933)` 和 `Whitened NCM (0.8867)`。差距主要体现在更具区分性的 `eurosat_fgvc` 上。这表明性能提升并不能完全由单层 frozen-feature 的线性读出或度量校准解释，多层级表示读取仍然带来额外收益。

### 建议改稿动作

- 在主文实验或附录新增一张 focused baseline table，至少包含：
  - `Zero-shot`
  - `SAV`
  - `Whitened NCM`
  - `Ridge Probe`
  - `RSEv2`
- 讨论里加一句：
  - `These baselines reduce the concern that HiRe’s gains stem only from a better classifier on a single frozen layer.`
- 不要把这组实验写成“HiRe 已经击败所有 possible frozen-feature baselines”；只能说它击败了本次新增的两个简单而合理的替代方法。

### 可直接放进实验段落的总结

To test whether the gain comes merely from a better readout on a single frozen layer, we added two simple baselines on the final decoder-block features: Whitened NCM and a closed-form Ridge Probe. On the FGVC trio (`eurosat_fgvc`, `pets_fgvc`, `cub_fgvc`) with Qwen2-VL, HiRe/RSEv2 achieves `0.9067` mean accuracy, exceeding Ridge Probe (`0.8933`) and Whitened NCM (`0.8867`). The gap is clearest on `eurosat_fgvc`, suggesting that hierarchical reading provides benefits beyond a stronger single-layer classifier.

---

## 4. 支持集使用可能导致信息泄露

### 问题判断

这条 `不属于真正的信息泄露`，但 `公平性说明确实不够清楚`。

few-shot classification 使用带标签的 support set 来估计 prototype / covariance，本身就是标准做法。只要没有用 query label，也没有用 validation label，这就不叫 leakage。  
审稿人真正抓住的是：你没有把 protocol fairness 写清楚。

### 可用证据

- 当前实验本来就是 support-only fitting。
- 新增的 `Whitened NCM` 和 `Ridge Probe` 也走的是同一 support/query protocol，这一点反而有利于说明 protocol 是统一的。

### 建议回复

我们理解审稿人的担心，但这里并不存在 query-side label leakage。HiRe 与新增的 frozen-feature baselines 都仅使用带标签的 support examples 来拟合类原型或协方差统计量，不使用任何 query label 或 validation label。这与标准 episodic few-shot classification 的设定一致。修订版中我们会明确写出这一点，并强调所有 few-shot 方法使用相同的 support budget。

### 建议改稿动作

- 在 setup 里明确加三句话：
  - 所有 few-shot 方法都只使用 support set 的图像与标签。
  - 不使用 query/validation labels。
  - 所有方法在相同 support budget 下比较。
- 若要更稳，可补一句：
  - `Our goal is matched supervision budget rather than identical inductive bias across methods.`

---

## 5. 超参数与实现细节描述不充分

### 问题判断

这条 `成立，而且优先级很高`。

这是最像“真缺陷”的一条，因为它直接影响可复现性。  
尤其要注意的是：当前 paper 和 code 在 shrinkage 的描述上并不完全一致。

### 可用证据

`RSEv2` 当前实现中的关键事实如下：

- `normalize_features=False`
- `covariance_shrinkage=auto` 时使用的是 `OAS-like` analytical shrinkage，而不是简单一句 Ledoit-Wolf 就能完全概括
- 有 `min_shrinkage_alpha`
- 有 `covariance_floor`
- 使用 low-rank `Woodbury` 形式求逆
- 当逆不稳定时使用 `pinv` fallback
- 默认 `confidence_floor=0.0`
- 默认 `component_top_k=0`

这些都来自当前实现：

- `src/methods/rsev2.py`

### 建议回复

我们同意当前版本对实现细节描述不足。修订版将补充完整的实现说明，包括 shrinkage 系数在实践中的计算方式、是否进行特征归一化、协方差稳定化与求逆方式、以及默认的 component weighting / top-k 设置。我们也会将正文中的 shrinkage 叙述与实际实现对齐，避免 paper-code mismatch。

### 建议改稿动作

- 在 supplement 单独加一个 `Implementation Details` 小节。
- 明确写出：
  - `RSEv2` 是否做特征归一化
  - shrinkage 的具体公式
  - `covariance_floor`
  - `Woodbury` 逆与 `pinv` fallback
  - `component_top_k=0` 的默认设置
  - `confidence_floor=0.0`
- 如果你不准备改代码，就一定要改论文表述，使其精确匹配代码。

### 可直接补进附录的简写版本

For `RSEv2`, we do not normalize component features before Mahalanobis scoring. The regularized covariance uses an automatic shrinkage coefficient computed from support residuals, together with a minimum shrinkage floor and an additive covariance floor for numerical stability. Inversion is implemented in a low-rank Woodbury form, with a pseudoinverse fallback when needed. Unless otherwise stated, the default configuration uses `component_top_k=0` and `confidence_floor=0.0`.

---

## 6. 聚合机制缺乏理论依据

### 问题判断

这条 `部分成立`，本质上也是 `写法过强`。

`margin` 加权当然是 heuristic，但你并不是完全没有经验支撑。问题不在于它不能用，而在于不能把它包装成一个具有严格校准理论的门控机制。

### 可用证据

已有消融：

- `RSEv2 = 0.7225`
- `RSEv2-EqualWeight = 0.7108`

说明：

- `margin` 加权在经验上确实有帮助。
- 但这只能支持“empirically useful”，不能支持“theoretically calibrated confidence”。

### 建议回复

我们同意审稿人的观点：当前的 margin-based weighting 主要是一个经验性信号，而不是具有严格校准保证的概率置信度。我们将相应降低表述强度，把它明确定位为一个简单、无参数、经验上有效的 confidence proxy。与此同时，我们会保留相应消融结果，以表明它在实践中优于 uniform aggregation。

### 建议改稿动作

- 删除或弱化下面这类措辞：
  - `Soft Mixture-of-Experts`
  - `closed-form gating`
  - `temperature alignment` 如果写得太满也建议降调
- 保留经验解释，但写成：
  - `margin serves as a simple per-query confidence proxy`
- 在 ablation 表里显式保留 `EqualWeight`。

---

## 7. 对失败案例分析不足

### 问题判断

这条 `现在已经基本缓解`，但可以再补一点。

主文现在已经不再是“完全没分析”，因为已经有 BLINK 边界讨论，也已经承认 closed-set discrimination 和 open-ended reasoning 的适用范围差异。  
所以这条更像“分析深度还可再加强”，而不是硬伤。

### 可用证据

- 主文已经讨论 BLINK 弱项来自 spatial / relational reasoning。
- 主文已经把方法范围收到了 closed-set discrimination。

### 建议回复

我们感谢这一建议。修订版已经进一步明确 HiRe 的适用边界：它主要针对 closed-set discriminative tasks，而不是多步生成式推理。针对 BLINK，我们会把失败原因写得更明确，即这类任务往往依赖空间关系建模、跨区域推理或完整自回归解码链，而这些能力并不一定能由单步 representation reading 充分替代。

### 建议改稿动作

- 保留当前 BLINK 讨论。
- 如果版面允许，可以在附录再加一个：
  - `BLINK subtask breakdown`
  - 或 `2–4` 个典型失败案例截图/说明

---

## 8. 组件数量膨胀（4L）可能引入偏差

### 问题判断

这条 `成立一半`。

审稿人的担忧是合理的，因为如果你有 `4L` 个组件而又没有分析组件数的影响，确实容易被怀疑“是不是靠堆组件数拿分”。  
但你现在已经有一组很好的消融可以正面回应这件事。

### 可用证据

已有消融结果：

- `RSEv2-HeadOnly = 0.6917`
- `RSEv2-Top1 = 0.7000`
- `RSEv2-Top4 = 0.7025`
- `RSEv2-Top8 = 0.7058`
- `RSEv2-All = 0.7225`
- `RSEv2 = 0.7225`

这说明：

- 不是“组件越少越稳”。
- 也不是“随便堆一堆组件自然就过拟合”。
- 更准确的结论是：完整层级信息通常是有帮助的，而 aggressive top-k pruning 并不稳定受益。

### 建议回复

我们认同审稿人关于组件规模分析重要性的意见，因此在修订版中补充了 `Top1 / Top4 / Top8 / All` 以及 `HeadOnly` 的消融。结果显示，完整层级聚合在平均性能上优于单层或 aggressive top-k 裁剪；例如 `RSEv2-All = 0.7225`，高于 `Top1 = 0.7000`、`Top4 = 0.7025` 和 `Top8 = 0.7058`。因此，HiRe 的收益并不能简单归因于“组件数膨胀”，而更可能来自多层级表征之间互补信息的稳定整合。

### 建议改稿动作

- 在正文或附录加一张 `Top1/4/8/All + HeadOnly` 表。
- 删除“默认设置会自动排除大量低质量组件”的过强表述。
- 更稳妥的写法是：
  - `Using the full hierarchy is often robust, while aggressive top-k pruning is not consistently beneficial.`

---

## 9. 与 LoRA 的比较可能具有误导性

### 问题判断

这条 `成立`，但不需要补实验，关键是改 framing。

如果 LoRA 结果来自参考论文，而不是你们同协议复现，那它就不能被叫作 `upper-bound`，更不能写成“直接超过 LoRA”这种强比较。

### 可用证据

- 你已经确认这里的 LoRA 结果是参考论文的结果，不是本工作重跑。

### 建议回复

我们感谢审稿人的提醒。修订版将明确说明文中的 LoRA 数字来自 prior work reported results，仅作为一个参考点而非严格匹配协议下的上界比较。因此，我们会删除“upper-bound”等表述，并避免使用“exceeding LoRA”这类可能引发误解的措辞。

### 建议改稿动作

- 把 `upper-bound` 改成：
  - `reported reference result from prior work`
- 删除：
  - `exceeding LoRA`
  - `LoRA upper bound`
- 可加一句：
  - `We include LoRA only as a rough reference to prior reported performance, not as a controlled comparison under identical tuning budgets.`

---

## 10. 缺乏真实开放场景任务的评估

### 问题判断

这条 `对 broad claim 来说成立`，对方法本身不算致命。

你当前实验主要还是 closed-set classification。  
因此问题不是“实验无效”，而是“结论外推过大”。  
尤其像 `read, don't write` 这种口号式表述，确实容易被打。

### 可用证据

- 主文已经承认：
  - closed-set discrimination 是更适合 HiRe 的场景
  - generative reasoning 仍然重要

### 建议回复

我们同意当前证据主要支持 closed-set discriminative settings，而不足以支撑更广泛的“改变与 LMM 交互方式”的强主张。因此修订版将进一步收紧结论范围，把贡献明确限定为 training-free hierarchical reading for closed-set multimodal classification，并将“read, don’t write”改写为一个有范围条件的经验性建议，而非普遍原则。

### 建议改稿动作

- 摘要、结论、limitations 全部统一收口到：
  - `closed-set few-shot classification`
- 弱化：
  - `read, don't write`
  - `shift in how we interface with generative multimodal models`
- 可替换为：
  - `For closed-set discriminative tasks, reading internal representations can be a strong alternative to relying solely on generation.`

---

## 11. 未分析计算与内存开销

### 问题判断

这条 `部分成立`，但可以不用新实验也进行较强回应。

审稿意见的表述是“存储所有中间表示尤其是 head-level 可能带来显著内存开销”。  
这个说法不完全准确，因为当前实现并不会在拟合后永久保存所有原始中间激活。

### 可用证据

从当前实现看，`RSEv2` 在拟合后保留的是每个组件的：

- 类中心 `centroids`
- 残差低秩投影 `residual_projection`
- `Woodbury` 逆矩阵 `woodbury_inv`

也就是说：

- 不是把所有训练样本的所有中间激活都永久缓存下来。
- 推理时也是单次 forward 抽取 query 的层级表示，打分后即可释放。

### 建议回复

我们感谢审稿人的提醒。当前版本确实缺少对资源开销的明确讨论。我们会在修订版中补充说明：HiRe 在训练后并不会保留所有原始中间激活，而是仅保留每个组件的类中心和一个低秩协方差状态，因此其额外存储成本主要来自 per-component class statistics，而非完整激活缓存。我们也会把这一点写入方法和附录，并将内存开销作为实际部署时需要考虑的因素之一。

### 建议改稿动作

- 在方法或附录中明确说明 fitted-state 的内容。
- 强调：
  - 不保存全部 raw activations
  - 只保存 per-component statistics
- 如果你后续想再加强，可补一小段 analytic complexity：
  - per component state roughly scales with `O(Cd + Nd + N^2)`
  - 其中 `C` 是类别数，`N` 是 support size，`d` 是特征维度

### 更稳的论文写法

After fitting, HiRe stores per-component class centroids together with a compact covariance state, rather than caching all raw intermediate activations. Thus, the added memory cost comes from the per-component statistics used for readout, which we now discuss explicitly as a practical trade-off of the method.

## 三、建议你在论文中优先完成的修改顺序

如果时间有限，建议按下面顺序改：

### 第一优先级

- 收紧 `DPI` 和 `Gaussian optimality` 的理论口径。
- 修复 `RSEv2` 实现细节描述不充分的问题。
- 改掉 `LoRA upper-bound` 和 `read, don't write` 这类高风险表述。

### 第二优先级

- 把新增的 `Whitened NCM + Ridge Probe` baseline 表补进实验。
- 把 `Top1/4/8/All + HeadOnly + EqualWeight` 消融补进正文或附录。

### 第三优先级

- 明确 support-only protocol fairness。
- 在 BLINK 处再补 1 段失败机制说明。
- 在附录补一段 resource / fitted-state discussion。

## 四、最安全的整体论文定位

修完之后，整篇论文最安全也最有说服力的定位应该是：

- HiRe 是一个 `training-free` 的层级表示读取框架。
- 它主要面向 `closed-set multimodal classification / few-shot label matching`。
- 它的主要贡献不是“普遍替代生成”，而是：
  - 在无需梯度更新的前提下，
  - 从 LMM 的层级内部状态中读取判别信息，
  - 并在多个 closed-set 任务上提供稳定而强的表现。

## 五、一句话版本的总回复思路

如果你想把整套回复压缩成一个总述段落，可以用下面这个思路：

我们感谢审稿人提出的系统性意见。修订版将重点完成四类改进：第一，收紧理论表述，将 DPI 和 Gaussian optimality 明确降格为方法动机而非现实最优性证明；第二，补充关键实现细节，尤其是 shrinkage、归一化和数值稳定化；第三，新增 reviewer-requested 的简单 frozen-feature 基线（Whitened NCM 与 Ridge Probe），结果表明 HiRe 的收益不能完全由单层特征读出解释；第四，进一步收紧方法的适用范围，将结论明确限定在 closed-set discriminative tasks，并补充组件规模、失败案例和资源开销讨论。
