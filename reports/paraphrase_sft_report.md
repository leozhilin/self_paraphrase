# Paraphrase-Augmented SFT：方法与实验报告

---

## 1. 相关工作

我们的方法是 **token-matched 的确定性 paraphrase 增广**：对模型**自身** rejection-sampled 的答对 CoT 做
**保逻辑、保答案**的表层多变体改写，并跨 **text / chart / multimodal(VLM)** 验证其改善 OOD 泛化与缩短链长。
据此把相关工作分为三类：**同期工作（concurrent）/ trace 改写与自采样谱系（prior）/ 奠基 baseline（prior）**。
我们与所有近邻工作的区分始终落在三把楔子上：**(i) token-matched raw-vs-paraphrase 消融**（干净分离"措辞更多样"与"数据更多"）、
**(ii) 跨 text/chart/VLM 三域**、**(iii) self-rollout 无 teacher**（不蒸馏、不改逻辑）。

### 1.1 同期工作（concurrent，首发 ≥ 2025-09）

> 标 ⚠️ 的为**最接近的近重复**，正文须显式区分。

| 工作 | 相似点 | 关键差异（vs 我们的三把楔子）| 链接 |
|---|---|---|---|
| ⚠️ **Shape of Thought** | **直接用 LLM paraphrase 已有正确 CoT 再 SFT**，验证 distribution-alignment（"shape > correctness"）| 头条是 **teacher 蒸馏 + 错答 trace 也有用**，paraphrase 仅次要 probe；**无 token-matched 消融、无 OOD/短链目标，仅 text/code（MATH/GSM8K/Countdown/MBPP）无 chart/VLM** | [arXiv:2512.22255](https://arxiv.org/abs/2512.22255) |
| ⚠️ **In Their Own Words (RSD)** | trace 改写 + rejection sampling + 保逻辑 + 小 Qwen3 | **token-level teacher→student 蒸馏** vs 我们 **self-rollout 无 teacher** 的确定性多变体；无 token-matched；text-only | [arXiv:2509.22230](https://arxiv.org/abs/2509.22230) |
| ⚠️ **DART: Learning to Adapt SFT Data** | transform-then-SFT 已有正确 trace + Qwen3 + 泛化目标 | 学一个 **RL mapper 重构逻辑**（1 版本/例）vs 我们确定性表层 paraphrase（n 变体）+ token-matched；0.5–0.6B 小模型，无 chart/VLM | [arXiv:2605.26924](https://arxiv.org/abs/2605.26924) |
| ⚠️ **TMS (Trajectory-Mixed Supervision)** | 同诊断（SFT mode-collapse / forgetting）、reward-free、用模型自身输出、GSM8K/MATH | diversity 来自**历史 checkpoint 时间蒸馏** vs 我们 paraphrase；无 rejection-of-correct、无 paraphrase（与我们**可并用**）| [arXiv:2602.03073](https://arxiv.org/abs/2602.03073) |
| **Less is More Tokens** | "改写正确 CoT 求短链" | **1→1 难度感知压缩 + teacher 蒸馏 + SFT+DPO** vs 我们 1→n 多样化 self-mine；**math-only（非多模态）** | [arXiv:2509.05226](https://arxiv.org/abs/2509.05226) |
| **Self-Distillation (SDFT)** | 同动机：贴近自身分布防遗忘 | reverse-KL 蒸馏（demo-conditioned teacher）vs 我们 plain SFT on 自身正确 trace 的 paraphrase | [arXiv:2601.19897](https://arxiv.org/abs/2601.19897) |

**其他正交同期工作**（loss / weight / question 级，与我们 data-级 trace-paraphrase 正交，一行带过）：
RPD [2510.26122](https://arxiv.org/abs/2510.26122) · P-TTS [2510.09599](https://arxiv.org/abs/2510.09599) · SED-SFT [2602.07464](https://arxiv.org/abs/2602.07464) · RoParQ（paraphrase *问题*，防混淆）[2511.21568](https://arxiv.org/abs/2511.21568) · More Data or Better Data? [2510.07169](https://arxiv.org/abs/2510.07169) · Info-Preserving Reformulation [2510.11545](https://arxiv.org/abs/2510.11545) · Weight Ensembling [2504.10478](https://arxiv.org/abs/2504.10478)。

### 1.2 trace 改写 / 自采样谱系（prior）

| 工作 | 相似点 | 关键差异 | 链接 |
|---|---|---|---|
| **LS-Mixture SFT** (Yu et al., 2025) | 用 LLM 对已有 CoT 做 **structure-preserved rewriting**，再 SFT | 动机是压缩 overthinking（long→short mixture），数据来自 R1 蒸馏；非 self-rollout，也不强调 OOD / 表述多样性 | [arXiv:2505.03469](https://arxiv.org/abs/2505.03469) |
| **RFT** (Yuan et al., 2023) | 自采样 + 答案过滤 → 仅正样本 SFT；指出 distinct reasoning paths 助泛化 | 按 equation-list **去重保留*不同逻辑*解**（与表层 paraphrase 正相反）；**不做 paraphrase** | [arXiv:2308.01825](https://arxiv.org/abs/2308.01825) |
| **RAFT** (Dong & Xiong et al., 2023, TMLR) | reward-ranked / best-of-n 拒绝采样 → SFT | 只取 top-1 丢弃其余，**从不生成 paraphrase 变体**（注：arXiv 2504.11343 为另一篇 2025 同缩写论文，勿混）| [arXiv:2304.06767](https://arxiv.org/abs/2304.06767) |
| **Retro-Search** (Lu et al., 2025) | 对已有 reasoning trace 做 **retrospective revision** 后用于蒸馏 / SFT | 改的是**逻辑路径**（探索未走分支、剪枝冗余），不是 surface paraphrase；面向效率与路径质量 | [arXiv:2504.04383](https://arxiv.org/abs/2504.04383) |
| **DCoT** (Puerto et al., ACL 2025) | 训练模型生成**多条不同 CoT**，提升推理与自改进 | inference-time 生成不同**逻辑**链再选答案，不是对已有正确链做同逻辑表述改写 | [arXiv:2407.03181](https://arxiv.org/abs/2407.03181) |
| **rStar-Math** (Guan et al., ICML 2025) | 自进化 + rejection sampling + verified traces → SFT | 用 MCTS + step-level Q-value 选**新路径**；**无 paraphrase 步骤**，也不做 token-matched raw vs rewrite 消融 | [arXiv:2501.04519](https://arxiv.org/abs/2501.04519) |

### 1.3 奠基 baseline（prior，含防混淆引用）

| 工作 | 相似点 | 关键差异 | 链接 |
|---|---|---|---|
| **STaR** (Zelikman et al., NeurIPS 2022) | self-rollout + answer-filter → SFT 的鼻祖 | 单 rationale/题，**从不 paraphrase**，无 token-matched 消融（含 rationalization 可能改逻辑）| [arXiv:2203.14465](https://arxiv.org/abs/2203.14465) |
| **MetaMath** (Yu et al., ICLR 2024) | math-SFT 数据增广经典 | rephrase **问题** + 加 forward/backward **新逻辑**（外部 GPT）；我们改 **trace** 表层保逻辑 | [arXiv:2309.12284](https://arxiv.org/abs/2309.12284) |
| **WRAP** (Maini et al., 2024) | paraphrase 数据增广源头 | paraphrase 通用 web 文档做**预训练**多风格；无 answer-filter（与我们多样性隔离思路平行）| [arXiv:2401.16380](https://arxiv.org/abs/2401.16380) |
| **Rephrase and Respond** (Deng et al., 2023) | paraphrase + reasoning | paraphrase **问题**、推理时 prompting、**零训练** vs 我们 trace 级训练数据合成 | [arXiv:2311.04205](https://arxiv.org/abs/2311.04205) |
| **ReAlign** (Fan et al., ACL 2024 Findings) | 最接近的 "reformat-not-augment 再 SFT" 先例 | 用 LLM+检索证据 reformat **外部** response（可改 / 加内容）vs 我们纯表层 paraphrase 自挖正确 trace 的多变体 | [arXiv:2402.12219](https://arxiv.org/abs/2402.12219) |
| **RefAug** (EMNLP 2024) | math-SFT 增广对照 | 追加 reflection（**新路径、拉长** trace）vs 我们保逻辑、缩短 | [arXiv:2406.12050](https://arxiv.org/abs/2406.12050) |

> *引用核验说明*：1.1 中 Shape of Thought / In Their Own Words / DART / TMS / Less is More Tokens 及 RFT、RAFT 的 arXiv id 与归属已人工复核；其余 2026 年新条目的 id 建议定稿前再扫一遍 arXiv 确认。


---

## 2. 方法

### 2.1 动机

1. **更好的 OOD 泛化**：传统 SFT 往往能让 ID 性能大幅提升，但常会牺牲 OOD，甚至造成灾难性遗忘。paraphrase 本质上是在
   **改写思维**——就像人类做数学题，可以有多种不同思路通向同一个正确答案；改写就是在促进这件事。
   用多种不同方式解题，提升的是**推理能力**，而不仅仅是模仿、记忆某一条固定说法或答案。

2. **Token 开销**：长链推理能提精度，但推理时生成长 CoT 增加延迟与算力成本。paraphrase trace 通常
   比原始 rollout 更短；在**与 raw 相同的 token 预算**下，改写提供更多条、更短的正确推理样本——既
   回应「要不要思考」的效率顾虑，也用同预算对照隔离「多数据」与「多表述」的贡献。

3. **无 trace 数据训练**：推理 SFT 需要「问题 + 推理过程 + 答案」，但多数数据集只有 `(问题, 标准答案)`。
   本 pipeline 用 rollout 采样 + 答案过滤，从模型自身输出里筛出**答案正确**的 trace，为任意「只有答案」
   的数据集自造推理监督，不依赖人工 CoT。

### 2.2 总体流程

以基座模型 `Qwen3-4B-Instruct`（文本任务）/ `Qwen3.5-4B`（多模态）为例，pipeline 分四步：

```
(问题, 答案)
   │  ① Rollout 采样：模型对每题自采样 G=32 条 (推理+答案)
   ▼
原始 rollouts
   │  ② 正确性过滤 + token 预算采样 → raw trace 池
   ▼
raw.jsonl  ──③ 改写增强（Paraphrase）──►  paraphrase.jsonl
   │                                         │
   └──────────────── ④ SFT ──────────────────┘
                       │
                       ▼
            LoRA adapter（raw / paraphrase / vanilla）
```

| 步骤 | 脚本 | 说明 |
|---|---|---|
| ① Rollout | `01_sample_*_rollouts.py` | 用 vLLM 对每题采样 **G=32** 条完整「推理+答案」（temperature=0.7, top_p=0.8, top_k=20）|
| ② Raw manifest | `02_build_raw_manifest.py` | **只保留答案正确的 rollout** 作为 trace 源；按 num_correct/G 分 bin，token 预算 **480k**、非全对样本占比 42% 做 bin-aware 采样 |
| ③ Paraphrase | `03_generate_paraphrases.py` | 对每条 raw trace 生成 **n_per=2** 个改写版本（保持答案正确性校验）|
| ④ Paraphrase manifest | `04_build_paraphrase_manifest.py` | 过滤改写候选（答案不符/过短/重复），与 raw **token-matched**（同 480k 预算）|
| ⑤ SFT | `05_sft_train.py --condition all` | LoRA 微调（3 epoch, lr=2e-5），分别训出 raw / paraphrase / vanilla 三个 adapter |

### 2.3 三个对照条件（condition）

| condition | 训练数据 | trace 内容 | 作用 |
|---|---|---|---|
| **base** | —（原始模型，无 LoRA）| — | 未训练基线 |
| **raw** | 模型自采样的正确 rollout（480k token）| 完整自生成推理链 | 验证「自造 trace」可行 |
| **paraphrase** | 对 raw 的改写增强（与 raw token-matched）| 改写后的推理链 | **本方法**：验证改写增强的额外价值 |
| **vanilla** | 见下 | 仅 `Final Answer: <gold>`（chart/mllm）或人工 CoT（GSM）| 「只学答案 / 标准 SFT」对照 |

> **vanilla 的任务差异（重要）**：
> - chart / MLLM 的 vanilla = `Final Answer:<gold>`，**无推理**（隔离推理链贡献，`02d` / `02e`）。
> - GSM 的 vanilla = GSM8K 官方**人工 CoT**（剥离 `<<calc>>` 标记，全量 7473 题，`02c`），是文献常报的标准 SFT。
> - 注意 vanilla 用全量数据，**未与 raw/paraphrase 做 token 对齐**，对比时需说明。

### 2.4 关键设计

- **自造 trace（rejection sampling）**：以「答案正确」作为弱监督，从模型自身采样里筛出自洽推理链，
  不依赖任何人工 CoT 标注 —— 这是方法适用于「无 trace 数据集」的根本。
- **token-matched 对照**：raw 与 paraphrase 共享同一 480k token 预算（见下表），保证两者
  数据量等价，**改写增强的增益不是来自更多数据**。
- **统一输出格式**：所有 condition 都以 `Final Answer: <answer>` 结尾，下游用统一规则提取。

各任务 raw manifest 规模（token 预算均 480k，非全对样本占比 42%）：

| 任务 | trace 数 | 覆盖问题数 | tokens |
|---|---|---|---|
| GSM | 1808 | 1509 | 480k |
| chart | 2114 | 1972 | 480k |
| MLLM | 421 | 378 | 480k |

---

## 3. 实验

### 3.1 设置

- **基座**：Qwen3-4B-Instruct（GSM / chart）、Qwen3.5-4B（MLLM 多模态）
- **训练**：LoRA（r=16, α=32, dropout=0.05），3 epoch，lr=2e-5
- **评测**：vLLM 贪心解码（temperature=0），统一 `Final Answer:` 提取；MCQ 支持「字母↔选项全文」双向匹配；数值用 math-verify 容差匹配
- **训练域**：GSM=GSM8K train；chart=ChartQA train；MLLM=AI2D train（其余均为 OOD 评测集）

### 3.2 GSM（文本数学，9 个公认测试集）

| 数据集 (n) | base | raw | paraphrase | vanilla |
|---|---|---|---|---|
| gsm8k_test (1319) | 93.5 | 93.4 (-0.1) | **93.6** (+0.1) | 81.8 (-11.7) |
| gsm_symbolic (5000) | **91.8** | 91.5 (-0.3) | 90.7 (-1.1) | 75.3 (-16.5) |
| svamp (1000) | 95.5 | **95.6** (+0.1) | 95.4 (-0.1) | 81.1 (-14.4) |
| asdiv (301) | 98.3 | 97.7 (-0.6) | **98.7** (+0.4) | 93.7 (-4.6) |
| multiarith (180) | **100.0** | **100.0** (+0.0) | 99.4 (-0.6) | **100.0** (+0.0) |
| aqua_rat (254) | 85.0 | **87.4** (+2.4) | 87.0 (+2.0) | 66.9 (-18.1) |
| math500 (500) | **81.2** | 80.0 (-1.2) | 79.6 (-1.6) | 50.8 (-30.4) |
| mawps (520) | 68.5 | 70.4 (+1.9) | 69.0 (+0.5) | **76.2** (+7.7) ⚠️ |
| gsm_hard (1319) | 59.9 | 60.0 (+0.1) | **60.1** (+0.2) | 51.5 (-8.4) |


### 3.3 chart（图表问答，chartqa train → 多 benchmark 评测， 全量微调）

基座 **Qwen3-4B-Instruct**；**全参数微调（full fine-tuning）**，raw/paraphrase token-matched（各 1.44M token），vanilla 为全量 ChartQA train（仅 `Final Answer:`，不参与 token 对齐）。结果时间 2026-06-22，`results/chart/eval/`。

| 数据集 (n) | base | raw | paraphrase | vanilla |
|---|---|---|---|---|
| chartqa_test (2500) | 65.7 | 76.7 (+11.0) | 81.2 (+15.5) | **82.4** (+16.7) |
| plotqa (5000) | 59.8 | **63.6** (+3.8) | 63.1 (+3.3) | 54.2 (-5.6) |
| tabmwp (7686) | 93.2 | 95.5 (+2.3) | **95.6** (+2.4) | 67.2 (-26.0) |
| finqa (1147) | 48.1 | 62.2 (+14.1) | **69.8** (+21.7) | 27.7 (-20.4) |


### 3.4 MLLM（多模态，PGPS9K train → 多 benchmark 评测）

基座 **Qwen3.5-4B**；SFT 训练数据来自 PGPS9K train 自采样 rollout（raw 2134 条 / paraphrase 3768 条，token-matched），GRPO 为 PGPS9K train 上的强化训练（LoRA r=16，reward=format+accuracy）。下表为**统一评测**结果（2026-06-22，`results/mllm/eval_unified/`）：base/raw/paraphrase/vanilla/GRPO 五个模型在**同一 vLLM 实例、同一评测代码、同一解码配置**下评测，base 口径完全一致。

| 数据集 (n) | base | raw | paraphrase | vanilla | GRPO |
|---|---|---|---|---|---|
| pgps9k_test (1000) | 73.8 | 75.1 (+1.3) | **78.2** (+4.4) | 66.8 (-7.0) | 76.5 (+2.7) |
| mathverse (3940) | 47.0 | 48.5 (+1.5) | **50.5** (+3.5) | 40.1 (-6.9) | 49.5 (+2.5) |
| mathvision (3040) | 40.5 | 41.6 (+1.1) | 44.1 (+3.6) | 22.4 (-18.1) | **44.9** (+4.4) |
| ai2d_test (3088) | 83.0 | 84.1 (+1.1) | 84.9 (+1.9) | **85.8** (+2.8) | 83.2 (+0.1) |
| mmmu_pro (1730) | 58.0 | 58.4 (+0.4) | 60.8 (+2.8) | 53.1 (-4.9) | **61.3** (+3.4) |

> 统一评测消除了此前 SFT 与 GRPO 分两批跑导致的 base 不一致。paraphrase 在 pgps9k/mathverse 上为全场最优（含超过 GRPO），mmmu_pro 仅次于 GRPO；GRPO 在 mathvision/mmmu_pro 上最强。两种方法在数学/几何推理任务上各有所长且接近，均显著优于 raw；vanilla 仅在 ai2d 居首，其余任务（尤其 mathvision -18.1）灾难性退化。GRPO 另在 geometry3k/geoqa/hle 上评测，详见 `results/mllm/eval_grpo/`。

---

## 附：复现入口

| 任务 | 全量脚本 | 评测脚本 | 结果目录 |
|---|---|---|---|
| GSM | `gsm_full_retrain.sh` | `run_eval_unified.sh` | `results/eval/` |
| chart | `chart_full_retrain.sh` | `run_chart_pipeline.sh` | `results/chart/eval/` |
| MLLM | `run_mllm_full_gpu0.sh`（GPU0 高吞吐）| `06_eval_mllm_vllm.py` | `results/mllm/eval/` |
