# Defense 实际评估情况与测试协议

本文整理当前讨论确定的 defense 评估范围。与同目录 `Attack评估说明.md` 区分：不同防御目标使用不同效果指标，不给所有方法统一套用 M1–M7。本文是协议与结果现状记录，不代表 defense 自动测试入口已经实现。

## 实验范围

Defense 使用 Qwen2.5-72B-Instruct teacher、Qwen2.5-7B base student，查询预算 B=1000。生成型防御在 SeqKD、QEDKS、SODA 三种 attack 下测试，均与匹配的无防御 clean student 比较。

| 类别 | 方法 | 当前已有 | 确定的后续测试 |
| --- | --- | --- | --- |
| 查询流检测 | MMD、PRADA、SEAT | 两组查询流的批次级检测统计与原始 batch scores | 现阶段直接复用，不新增测试 |
| 抗蒸馏 | ADS、DOGe、Trace Rewriting | Defended student checkpoint、clean student checkpoint、训练产物 | Teacher、clean student、defended student 的 M1/M2 |
| 标记归因 | ADFP、GINSEW、Radioactivity | Checkpoint、protected/clean 检测报告及相关检测输出 | 复用标记检测；补 teacher、clean、protected 的 M1/M2 |
| 训练后谱系验证 | DuFFin | 暂未运行、暂无结果 | 派生/无关模型识别效果与验证成本；不把 M1/M2 设为必测 |

数据来源：


三个模型仓库提供六种生成型防御和对应 clean 基线。SeqKD 部分 `stage1_results_v1.json` 虽然存在，但其 M1/M2/M3 明确为 unavailable，不能把训练完成当作下游评估已完成。

## 一、Query traffic：现阶段结果已齐

按批次检测是当前协议的正常设计。每批 50 条查询；每组有 1000 条攻击查询，即 20 个攻击批次。正常数据中 800 条用于参考/校准，200 条用于独立测试，即 4 个正常测试批次。

SeqKD 与 SODA 使用相同固定查询，因此只保留一组固定查询检测实验，另有 QEDKS 查询组；不重复计算第三组作为独立证据。

| 查询组 | 检测器 | TP / 攻击批次 | FP / 正常批次 | TPR | FPR |
| --- | --- | --- | --- | --- | --- |
| SeqKD/SODA 固定查询 | MMD | 0/20 | 0/4 | 0% | 0% |
| SeqKD/SODA 固定查询 | PRADA | 1/20 | 1/4 | 5% | 25% |
| SeqKD/SODA 固定查询 | SEAT | 1/20 | 0/4 | 5% | 0% |
| QEDKS | MMD | 20/20 | 0/4 | 100% | 0% |
| QEDKS | PRADA | 0/20 | 1/4 | 0% | 25% |
| QEDKS | SEAT | 20/20 | 0/4 | 100% | 0% |

结果已保存在 `b1000/<attack>/<detector>/detector_report.json` 的 `external_metadata.metrics`，对应原始记录在 `<detector>/batch_scores.csv`。

论文报告“校准阈值下的批次级测试 TPR/FPR”，不改称 TPR@5% FPR。阈值校准使用 95 百分位，不等于测试 FPR 已固定在 5%；正常测试只有四批，误报率以 25 个百分点跳变。0/4 代表当前样本未误报，不代表真实误报率为零。这不否定已有结果：在当前查询构造与批次协议下，QEDKS 与固定查询组的检测表现差异明显。现阶段无需因为批次少就推翻统计或重跑。

现有批次 TPR 不等同于完整攻击流的检测率；本文不把首次可疑批次转换成已验证的在线 queries-to-detection 指标。若未来研究检测延迟或低 FPR，需要另外固定对应协议。

## 二、六种生成型防御：统一补 M1/M2

### 比较对象

- Teacher：无防御 Qwen2.5-72B-Instruct。
- Clean student：同一 attack、B=1000、无防御 Qwen2.5-7B student。
- Defended/protected student：同一 attack、B=1000、从防御 teacher 回答训练的 student。
- 原始 Qwen2.5-7B base：额外共享一份 M1 基线，用于判断 student 是否获得能力提升。

Clean、protected 和 base 不是同一对象。基线与数据应先核对查询、模型、预算和训练协议是否匹配；不能拿 Llama attack_b100 的 checkpoint 作 Qwen defense 基线。

### M1：六任务宏平均 ACC

沿用 Attack 确定的六任务与原有 rollout 答案解析方式：ARC-Challenge test、HellaSwag validation、MMLU all/test、TruthfulQA multiple_choice/validation（MC1）、WinoGrande winogrande_xl/validation、GSM8K main/test。

所有对象使用相同零样本题目模板、评估样本、答案解析器及生成协议；解析失败计错误。每任务分别计算 ACC，再等权宏平均，保留分项和失败数量。Teacher ACC 可跨防御与 attack 共享；base ACC 也只需计算一次。训练查询与测试题的重合检查仍需完成。

### M2：BERTScore

在同一批 held-out prompts 上生成 clean/protected student 回答，统一使用无防御 Qwen teacher 回答作参考，报告 baseline-rescaled BERTScore F1。Teacher 是参考对象，不需再引入 teacher-vs-self 作为有信息量的结果列。编码器与长文本处理协议应与 Attack 保持一致并记录。

已有 3000 条 held-out prompts 可以共享，但已经生成的 Llama-3.3-70B-Instruct teacher 回答不能用作 Qwen defense 的 reference。必须核对是否已有同批 Qwen teacher 回答；没有时生成一次，各组合复用。不能直接把训练阶段 defended transcript 当 held-out reference。

已新增 `evaluation/defense_eval/evaluate.py` 与 `runs/evaluation/` 的提交脚本。共享 reference 任务直接在 GPU 上生成无防御 Qwen teacher 的 held-out 回答、计算 teacher/base 六任务 ACC；student 阶段依赖 reference 成功完成，不读取 Llama teacher 回答。现有文件是否包含可替代的新协议 Qwen reference 尚未核对，因此默认生成并持久化到本次输出目录，后续运行按协议复用。

### 两组防御如何解释

ADS、DOGe、Trace Rewriting：关注相对 clean 的 student ACC 与 BERTScore 是否下降。M1 与 M2 分别衡量正确性和对原 teacher 的语义模仿，避免仅凭训练 loss 判断防御效果。

ADFP、GINSEW、Radioactivity：结合已有标记检测，关注 student 的能力和模仿效果。若相对 base 有能力提升、接近 teacher，但 protected 与 clean 的标记证据仍无法有效区分，可以支持“能力发生迁移，但标记未实现有效归因”的解释。若没有能力提升，仍可报告当前归因表现，但不能进一步声称成功提取能力却未继承标记。

本轮六方法补测限定 M1/M2，不自动增加 Rep-4、judge、PPL 等指标。正常用户受保护回答的效用是另一评估对象，本轮尚未决定新增该实验，不与 student ACC 混称。

## 三、已有标记检测：直接比较 protected 与 clean

已有 GTP、绿色比例、检测分数、p/z 及 clean 对照，可以直接分析，不是只有 p 值，也不需要先重新生成才能评价。每个 attack–defense 组合有一个 protected student 和一个匹配 clean student；1000 条 probe 不等于 1000 个独立模型。

### ADFP：GTP

| Attack | Protected GTP | Clean GTP |
| --- | --- | --- |
| SeqKD | 0.498258 | 0.498747 |
| SODA | 0.498091 | 0.497987 |
| QEDKS | 0.507868 | 0.506536 |

当前统计接近，没有清楚的指纹区分证据。

### GINSEW：绿色 token 比例

| Attack | Protected | Clean | 差值/百分点 |
| --- | --- | --- | --- |
| SeqKD | 48.260521% | 48.148637% | +0.111884 |
| SODA | 47.874753% | 48.021049% | -0.146296 |
| QEDKS | 52.648952% | 51.792450% | +0.856502 |

QEDKS 有较大的数值差异，但单次统计差异不自动等于稳定、可靠的模型归因能力。保留这个实测差值，不把它抹掉，也不凭该差值直接宣称有效归因。

### Radioactivity：绿色 token 比例与检测 p 值

| Attack | Protected 绿色比例 | Clean 绿色比例 | Protected p | Clean p |
| --- | --- | --- | --- | --- |
| SeqKD | 25.053079% | 25.043512% | 0.323586 | 0.354519 |
| SODA | 25.297825% | 25.233735% | 0.032147 | 0.073177 |
| QEDKS | 25.728280% | 25.710959% | 2.54172e-8 | 2.61075e-8 |

QEDKS 两者 p 都很小，但绿色比例接近，clean 也有偏离随机基准的信号；不能仅凭 protected 小 p 值宣称标记继承。SODA 中一个 p 小于 0.05、另一个大于 0.05，不等于两组差异显著。

p 是针对各自检测零假设的尾概率，不是无水印的后验概率，也不是信号强度。两组 p 值的差不是差异显著性检验。但已有绿色比例、GTP、score 和对照足够做描述性比较，现阶段不要求先做额外检验才能报告结果。只有进一步宣称差异显著或跨 seed 稳定时，才需要对应分析或重复实验。

当前结果中没有哪个组合已经证明可靠的模型级归因；这不等于证明所有条件下都无效，也不等于证明植入标记绝对没有任何统计影响。不因负面结果要求重跑。现有检测报告直接保留，结合后续 M1/M2 解释。

### SODA 运行说明的处理

SODA 模型卡称除 ADS 外，多数分支 DPO 更新较弱。该描述不代表已确认执行失败，也不能仅凭 loss 推断模型没学到能力。M1/M2 和 base/clean 对照用于核实结果，不因此预先要求重训或否定已有检测。

## 四、DuFFin：单独的训练后验证


默认测试派生模型与无关模型的识别效果（TPR/FPR），以及验证成本。阈值在独立校准数据上确定，效果在测试模型上测量，具体低 FPR 目标取决于负样本规模。若未来报告 TPR@固定 FPR，需具备足够独立负模型数据。

不把 M1/M2 设为 DuFFin 必测。若复用前述已评估 student，直接引用已有能力和保真度；研究能力与谱系验证关系属于可选扩展。

## 五、接下来的工作

### 一条命令提交补测

在集群仓库根目录执行：

```bash
```

自动提交两个阶段：reference 作业申请两张 B200，依次测试 Qwen teacher、base 并生成 teacher held-out reference；student array 的三个任务各申请一张 B200，分别处理 SeqKD、SODA、QEDKS，各测 clean 与六种 defended student。student 阶段以 afterok 依赖 reference。不会重新训练，也不重新运行水印或 query traffic 检测，不包含 DuFFin。

前置条件：research 环境已有 torch、transformers、accelerate、peft、datasets、PyYAML、huggingface_hub、bert-score。缺 BERTScore 时先执行 `python -m pip install bert-score==0.3.13`。默认直接 Transformers/PEFT 加载，无需 vLLM 服务。每个作业时限 168 小时；实际全量耗时须通过集群运行确认。

先验证小样本：

```bash
```

Smoke 使用每任务和 held-out 的前两条，不代表正式结果。默认与全量目录分离。HF 模型、数据、BERTScore 编码器缓存均放 STORAGE_ROOT 下；默认 prompts 为仓库 `evaluation/data/heldout/heldout_prompts.jsonl`，可随 Git 共享。可用 HELDOUT_PROMPTS_JSONL 覆盖，但同一运行目录的协议变化会报错。Qwen teacher 回答仍写入 STORAGE_ROOT 的 reference/teacher/heldout_outputs.jsonl，不加入 Git。

正式输出为 `${STORAGE_ROOT}/results/defense_eval/b1000/`，reference/ 存 teacher/base M1 和 teacher reference，seqkd/、soda/、qedks/ 各有 summary.json/csv。每份 summary 包含 student ACC、BERTScore、teacher/base ACC、相对 base 提升及相对 clean 的 ACC/BERTScore 降幅（按 0–1 分数差，不是百分比相对变化）。每组完整应有七行。逐题结果实时追加，重提同一协议跳过已生成样本；损坏的 JSONL 会明确报错，不静默跳过。

Teacher 使用其 instruct chat template。Qwen2.5-7B student 是 base 模型，base/clean/defended 统一使用 raw-text 输入，不强行套 instruct 对话模板。所有对象的 M1 题目模板、gold 和解析规则相同，原始提示到模型 token 的包装按模型类型记录。BERTScore 采用 roberta-large、英语 baseline rescaling；长文本处理沿用库行为。现阶段没有额外报告 teacher-vs-self 的 M2。

Checkpoint 来自三个公开模型仓库的 curated 最终目录及 clean 最终目录，不通过递归搜索任选中间 checkpoint。下载时固定模型仓库当前 commit，source.json 记录来源；验证每个 LoRA 的 base_model_name_or_path 为 Qwen/Qwen2.5-7B。仓库版本变化不允许在同一目录静默复用旧输出。

本地已进行语法、输入/协议与续跑逻辑检查，尚未进行集群 GPU 集成运行。若共享 reference 失败，student 依赖不会满足；先处理 reference 日志，再重新提交。

1. 保留并汇总两组 query traffic 现有批次 TPR/FPR。
2. 保留三种标记防御的 protected/clean 检测统计，不为负面结果重训。
3. 核对三种 attack 的 clean 和六种 defended checkpoint，统一 Qwen 模型与 B=1000 协议。
4. 核对或生成共享 Qwen teacher held-out reference、teacher/base M1 基线。
5. 实现六方法及 clean student 的六任务 ACC 和 held-out BERTScore 自动评估。
6. DuFFin 数据就绪后单独评估，不阻塞前述工作。
