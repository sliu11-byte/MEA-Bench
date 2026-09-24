# Counter 评估说明

## 一、评估目标与范围

Counter 评估回答：DIPPER 或回译能否削弱训练回答中的标记，并使最终 student 不再明显携带标记，同时保住 student 的能力和对 teacher 的模仿效果？

当前运行脚本包含两种反制方法：DIPPER、Translation（回译）；对应三种标记防御：ADFP、GINSEW、Radioactivity。实验预算为 B=1000。脚本支持 SeqKD、LoRD、SODA、GAD、QEDKS、Model Leeching；实际汇总以已完成实验为准，不把脚本支持范围写成已经完成的结果。

本文件记录当前精简评估方案。SeqKD B=1000 已补本地结果评估入口，不自动提交任务。

## 二、三组对照

| 实验组 | Student 的训练回答 | 作用 |
| --- | --- | --- |
| clean | 无防御 teacher 回答 | 没有植入标记时的能力、模仿效果和检测分数基线 |
| defense-only | 带标记的 teacher 回答，未做反制 | 反制前的能力、模仿效果和标记信号 |
| counter | 带标记回答经过 DIPPER 或回译处理 | 反制后的能力、模仿效果和标记信号 |

这里的 clean 是“未使用标记防御”，不是未经训练的 base 模型。三组 student 都是攻击训练的结果。Clean 可作植入标记的阴性对照，但仍是 teacher 的派生模型，不能当作 DuFFin 谱系检测中的无关模型。

同一个防御与攻击组合中，三组检测应使用同一套防御 artifacts、密钥及 probe 协议。对于固定训练查询的攻击，DIPPER 与回译使用相同原始 defense-only transcript；在线自适应查询攻击则按现有在线流程执行，不声称三组查询完全相同。

DIPPER 与回译共用同一组 clean 和 defense-only checkpoint，因此这两组 detector 输出也必须只生成一次并由两个分支共享。水印 detector 的生成固定随机种子；不能让两个分支分别以 `temperature=0.7` 重新采样 baseline，也不能把两次随机结果平均后当作独立重复。只有 counter/Rewritten student 的检测结果属于各自分支。

## 三、核心指标

| 指标 | 评估问题 | 需要的输入 |
| --- | --- | --- |
| M1：ACC | 反制后 student 的任务能力是否受损？ | 三组最终 checkpoint、M1 专用测试题及标准答案 |
| M2：BERTScore | 反制后 student 是否仍能模仿无防御 teacher？ | 三组 student 在共享 held-out 上的回答、对应无防御 teacher 的参考回答 |
| 标记检测分数（M4/M5） | 反制是否削弱最终 student 的标记信号？ | 三组 detector 报告及对应防御 artifacts/probes |

### M1：ACC

沿用已经确定的六任务协议：ARC、HellaSwag、MMLU、TruthfulQA MC1、WinoGrande、GSM8K。主表报告六任务宏平均 ACC，逐任务结果保留在附表或结果文件中。

Clean、defense-only、counter 使用相同测试题和评分方式。重点比较 counter 相对 defense-only 的能力变化，同时参考 clean 水平。原始 base ACC 可作共享背景基线，不增加新的主指标。

### M2：BERTScore

沿用 BERTScore 协议：roberta-large，rescale_with_baseline。三组 student 使用同一份 held-out prompts，并与同一个无防御 teacher 的参考回答比较。

共享 prompts 位于 `evaluation/data/heldout/heldout_prompts.jsonl`，共 3000 条。M1 专用测试题与 M2 held-out 是两套数据。

参考回答必须与实验实际 teacher 一致，并核对 prompt ID 和文本。不能只因复用了相同 prompts，就将其他 teacher 的回答用作 reference。M2 测试对象是最终 student 的输出，不是 DIPPER/回译改写后的训练回答本身。

### 标记检测分数：反制前后比较

不同防御保留其对应的一种主要检测统计，不跨方法比较绝对分值：

| 防御 | 主统计 | 辅助信息 |
| --- | --- | --- |
| ADFP | GTP | 保留原 detector 报告与检测样本数 |
| GINSEW | 绿色比例 | 保留原 detector 报告与检测样本数 |
| Radioactivity | 绿色比例 | p 值作为辅助，保留检测样本数 |

核心比较是 defense-only 与 counter 的变化，以及两者分别距离 clean 基线多远。P 值不是信号强度，不通过比较两个 p 值大小直接判断反制效果。

若 defense-only 本来就与 clean 接近，应报告该设置下没有观察到清晰的标记区分，不能仅凭 counter 也接近 clean 就称反制成功。现有三组检测统计仍可直接用于描述与比较，不因结果为阴性就要求重新训练或检测。

仓库 M5 提供反制前后 retention、loss、evasion/removal rate 等计算。本轮不将这些派生量全部加入主表。若后续报告阈值化移除率，应使用反制前确定的固定检测阈值，不在反制后重新调阈值；单个 clean student 也不能用于可靠估计模型级 FPR 或 ROC-AUC。

## 四、已有结果与待补工作

### SeqKD B=1000 本地补测入口

```bash
mkdir -p logs
# 分别确认输入检查和小样本成功后，再执行正式提交
```

直接使用 STORAGE_ROOT 下的 counter_baselines、countermeasures 和已有 Llama held-out teacher reference。评估 10 个最终 checkpoint 的 ACC、BERTScore，复用标记检测报告；不重新训练、不从 HF 下载实验结果、不加载 teacher。默认一张 B200、128G 内存、8 核 CPU。

结果位于 `$STORAGE_ROOT/results/counter_eval/seqkd_b1000/`。models_summary 包含 10 个模型的能力与输出相似度；summary 包含六组实验各自的三组对照，共 18 行，并保存原始检测字段。同一 defense 的 DIPPER 与回译行必须引用完全相同的 clean、defense-only 检测结果。运行参数和完成标准见 [评估入口 README](../runs/evaluation/README.md)。

旧结果曾让两个 counter 分支分别随机生成 baseline detector 的 student outputs，导致相同 checkpoint 出现不同 Clean/Protected 数值。修复时运行：

```bash
```

该任务固定 seed，只重跑三种 defense 的 Clean/Protected 共 6 次检测，保留两种 Rewritten 结果，并同步更新两份 comparison report。若 M1 v2 已完成，还会直接刷新最终 summary，不重跑 ACC 或 BERTScore。

现有 Counter 运行流程会检测 clean、defense-only、counter 三组 student，并写出三组 detector 报告和 `comparison_report.json`。实际已完成数据应检查这些文件是否齐全、对应 checkpoint 是否存在，以及检测协议是否一致。

| 内容 | 当前处理 |
| --- | --- |
| 三组标记检测 | 优先复用已完成实验中的 detector 报告与 comparison_report.json |
| 三组 M1：ACC | 需要补测最终 student checkpoint |
| 三组 M2：BERTScore | 需要补生成 held-out student 回答并评分；匹配的 teacher reference 可复用 |
| M3：输出质量 | 本轮不作为必测主指标 |
| M6：成本 | 可记录额外反制 GPU-hours，暂不作为必测主指标 |

成本如需报告，应区分反制改写、student 训练、基线准备和评估检测，避免将含共享基线的整作业时长直接称为纯反制成本。

运行结构见 `runs/counter/README.md`。默认输出：

```text
$STORAGE_ROOT/outputs/countermeasures/<attack>_b1000/<countermeasure>/<defense>/
```

主要检测文件位置：

```text
comparison_report.json
detector/<defense>/
baseline_detection/clean/detector/<defense>/
baseline_detection/defense_only/detector/<defense>/
```

Clean 和 defense-only checkpoint 的共享存储位置：

```text
$STORAGE_ROOT/outputs/counter_baselines/<attack>_b1000/clean/shared/<config-hash>/
$STORAGE_ROOT/outputs/counter_baselines/<attack>_b1000/defense_only/<defense>/<config-hash>/
```

## 五、论文展示

主表保留三个结果列：ACC、BERTScore、对应标记检测分数。按攻击方法与标记防御分组，展示 clean、defense-only、DIPPER counter、回译 counter；两种反制共享的基线无需重复计算。

理想的反制结果是：相较 defense-only，counter 标记分数向 clean 基线靠近，而 ACC、BERTScore 没有明显下降。若标记减弱但能力或模仿效果同时下降，则报告反制的效果与代价；若反制前信号就不清晰，则如实展示三组对照，不夸大移除结论。
