# Attack 实际评估协议

本文记录当前已确定的 attack 评估范围，以及 budget100 的实际运行方式。只评估 attack，不包含防御的归因、鲁棒性或在线检测指标。

## 四个核心指标

| 维度 | 主指标 | 方向 | 数据与解释 |
| --- | --- | --- | --- |
| M1 能力 | 六任务宏平均 ACC | 越高越好 | 标准任务公开带标签评估 split，生成并解析最终答案 |
| M2 保真度 | BERTScore F1 | 越高越好 | 3000 条 held-out 的 teacher/student 回答，采用 baseline-rescaled F1 |
| M3 生成退化 | Rep-4 | 越低越好 | 同一批 student 回答的四元组重复比例 |
| M6 资源成本 | Allocated GPU-hours | 越低越好 | 已完成攻击作业的分配 GPU 数乘作业运行时长 |

默认不运行 ROUGE、MAUVE、cross-PPL、oracle-PPL、judge。查询预算是实验约束，实际请求计数用于预算核验，不作为主表新增指标。

## M1 的任务协议

| 任务 | Hugging Face 来源 | 子集 | 评估 split |
| --- | --- | --- | --- |
| ARC-Challenge | allenai/ai2_arc | ARC-Challenge | test |
| HellaSwag | Rowan/hellaswag | 默认 | validation |
| MMLU | cais/mmlu | all | test |
| TruthfulQA | truthfulqa/truthful_qa | multiple_choice，MC1 | validation |
| WinoGrande | allenai/winogrande | winogrande_xl | validation |
| GSM8K | openai/gsm8k | main | test |

沿用 `evaluation/tasks/adapters/` 的题目模板与答案解析器，不使用选项 likelihood 或 TruthfulQA MC2。零样本、chat template、greedy decoding、评估 seed 42；选择题最多 32 个新 token，GSM8K 最多 512 个。解析失败计错误，不从 ACC 分母排除。每任务计算正确数/总题数，六任务等权宏平均；不是所有题目的微平均。分任务结果和解析失败数同时保存。

这些专用测试数据不由攻击输出产生，也不需要与训练查询池来源一一对应。脚本固定 split 并保存数据 fingerprint。仍需额外完成训练查询与能力测试题的重合审计；选择官方评估 split 本身不等于完成去重检查。原 M1 配置有 13 个任务，本入口只选择上述六个，不改动其他实验的配置。论文尚需补入这套能力测试协议。

## M2 / M3 的 held-out 数据

默认读取仓库 `evaluation/data/heldout/heldout_prompts.jsonl`（3000 条，约 1.37 MiB，可随 Git 共享）；Llama teacher reference 仍读取 `${STORAGE_ROOT}/outputs/heldout_queries/heldout_teacher_outputs.jsonl`。构建规则为 100000 查询池中跳过最大训练前缀 10000，选择其后三个千条区块，即 3000 个 prompts。共享文件从本地原始池重建，排序后 ID 哈希与已完成 teacher 作业一致；实际评估还会检查 teacher reference 的逐条 prompt 文本。现有 teacher 日志报告 Llama-3.3-70B-Instruct 生成 3000 条、失败 0 条。

Student 使用攻击最终 LoRA 的原始 base model，greedy decoding，最多 1536 新 token。生成前校验 prompt ID 与 teacher prompt 文本。M2 使用 `roberta-large`、英语 baseline rescaling，指标衡量与 teacher 的语义相似，不代表正确性。编码器具有有限上下文，BERTScore 的长文本处理遵循该库行为，不宣称覆盖每条长回答的全部 token。

Rep-4 为每条回答 `1 - unique_4grams / total_4grams` 后取宏平均。使用已有 Unicode/CJK 分词规则。少于四个 token 的回答不进入 Rep-4 平均分，报告其数量与比例；空回答另外计数。全为短回答时 Rep-4 为 null，不伪造为零。

## M6 的真实历史记录


| Attack | Job ID | 时长/秒 | B200 数量 | Allocated GPU-hours |
| --- | --- | --- | --- | --- |
| SeqKD | 41615096 | 142 | 1 | 0.039444 |
| LoRD | 41903837 | 5181 | 1 | 1.439167 |
| Model Leeching | 41612821 | 462 | 3 | 0.385000 |
| QEDKS | 41610488 | 1350 | 3 | 1.125000 |
| SODA | 41909181 | 783 | 2 | 0.435000 |
| GAD | 41912777 | 4529 | 2 | 2.516111 |

公式：`allocated_gpus * elapsed_seconds / 3600`。GPU 型号键和总数键不能重复累加。这是整个攻击作业的资源分配时，不是实际 GPU 忙碌时间，也不是纯训练时长。包含作业内的加载、等待、生成、训练等环节；不包含本次 evaluation。

重要限制：Model Leeching、QEDKS 作业内包含本地 teacher 服务；其他作业使用外部 teacher 或共享 transcript，外部服务及 transcript 构建成本不在这些数中。因此该列只标作“攻击作业分配 GPU-hours”，不能称为公平统一的端到端总计算成本。每份成本报告保留 teacher_scope。旧 stage1 文件的 gpu_hours=0 来自函数默认参数，本入口不采用它。

## 一条命令运行 budget100

在集群仓库根目录执行：

```bash
```



先做小样本验证：

```bash
```

Smoke 默认输出到独立的 `b100_smoke_2` 目录，不能当正式结果。正式运行不设置 EVAL_LIMIT/EVAL_ATTACK。每完成一条生成即写入 JSONL，重复提交同一协议会跳过已完成 ID。协议变化会拒绝复用，需更换 ATTACK_EVAL_OUTPUT_ROOT。若作业中断留下损坏 JSONL，脚本报错而不静默忽略，需要检查末行。

输出包括每个 attack/run 的 M1 逐题预测、held-out student 回答、四项指标 JSON 和运行协议。顶层 `summary.json` / `summary.csv` 直接包含 acc、bertscore、rep4、allocated_gpu_hours 四个数值；随每种方法完成更新。完整结果须有六种方法且 smoke=false。

本地仅能完成语法和协议逻辑检查；实际权重下载、GPU 生成与 BERTScore 需要集群 smoke 验证。Teacher/base 的能力基线尚未包含在本次 attack-only 入口中，不影响 student ACC 计算，但论文若报告能力提升或保留比例，需另行测试基线。
