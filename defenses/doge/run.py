import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from defenses.core.constants import DEFENSE_ROLE, DEFENSE_TYPE
from defenses.core.cost import build_cost
from defenses.core.io_utils import ensure_dir, read_jsonl, write_jsonl
from defenses.core.manifest import DefenseManifest


def _pick(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


def _run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


class DOGeTrainingDataset:
    """
    DOGe defensive-training dataset.

    Expected input records:
      {
        "prompt": "...",
        "teacher_response": "..."
      }

    The prompt is masked from LM loss; only teacher response tokens are targets.
    """

    def __init__(self, path: str, tokenizer, max_length: int = 1024):
        import torch

        self.examples: List[Dict[str, torch.Tensor]] = []

        for record in read_jsonl(path):
            prompt = _pick(record, "prompt", "query", "question", "instruction")
            response = _pick(record, "teacher_response", "response", "answer", "output", "completion")
            if prompt is None or response is None:
                raise ValueError(
                    "DOGe training records require prompt/query and teacher_response/response fields."
                )
            prompt = str(prompt)
            response = str(response)

            prompt_messages = [
                {"role": "user", "content": prompt},
            ]

            full_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_text = tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                prompt_text = f"### User:\n{prompt}\n\n### Assistant:\n"
                full_text = prompt_text + response

            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]

            enc = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            labels = list(input_ids)
            prompt_len = min(len(prompt_ids), len(labels))

            for i in range(prompt_len):
                labels[i] = -100

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

        if not self.examples:
            raise ValueError(f"No DOGe training examples found in {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_batch(batch, pad_token_id: int):
    import torch
    import torch.nn.functional as F

    max_len = max(x["input_ids"].shape[0] for x in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for x in batch:
        n = x["input_ids"].shape[0]
        pad = max_len - n

        input_ids.append(
            F.pad(x["input_ids"], (0, pad), value=pad_token_id)
        )
        attention_masks.append(
            F.pad(x["attention_mask"], (0, pad), value=0)
        )
        labels.append(
            F.pad(x["labels"], (0, pad), value=-100)
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }


def freeze_for_doge(teacher_model, proxy_model):
    # Proxy student is always frozen.
    for p in proxy_model.parameters():
        p.requires_grad = False

    # Freeze the entire teacher first.
    for p in teacher_model.parameters():
        p.requires_grad = False

    # DOGe v1 trains only the final LM/output head.
    lm_head = teacher_model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError(
            "DOGe requires a causal LM exposing an LM/output head."
        )

    for p in lm_head.parameters():
        p.requires_grad = True


def doge_loss(
    teacher_model,
    proxy_model,
    batch,
    anti_kd_coef: float,
    kd_temperature: float,
):
    import torch
    import torch.nn.functional as F

    teacher_outputs = teacher_model(**batch)

    with torch.no_grad():
        proxy_outputs = proxy_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    lm_loss = teacher_outputs.loss

    vocab_size = min(
        teacher_outputs.logits.shape[-1],
        proxy_outputs.logits.shape[-1],
    )

    teacher_logits = teacher_outputs.logits[..., :vocab_size].float()
    proxy_logits = proxy_outputs.logits[..., :vocab_size].float()

    # Causal LM logits at position t predict token t+1.
    # Shift logits/labels so the anti-KD mask is aligned with
    # the same response tokens used by the LM objective.
    teacher_logits = teacher_logits[:, :-1, :]
    proxy_logits = proxy_logits[:, :-1, :]
    shifted_labels = batch["labels"][:, 1:]

    # Only assistant/response target tokens participate in anti-KD.
    valid_mask = shifted_labels != -100

    teacher_logits = teacher_logits[valid_mask]
    proxy_logits = proxy_logits[valid_mask]

    if teacher_logits.numel() == 0:
        raise RuntimeError("No valid response tokens available for DOGe KL loss.")

    #
    # Matches the official DOGe direction:
    # KL(P_teacher || P_proxy)
    #
    kd_loss = F.kl_div(
        F.log_softmax(proxy_logits / kd_temperature, dim=-1),
        F.softmax(teacher_logits / kd_temperature, dim=-1),
        reduction="batchmean",
    )

    loss = lm_loss - anti_kd_coef * kd_loss

    return loss, lm_loss, kd_loss


def save_doge_checkpoint(
    teacher_model,
    tokenizer,
    output_dir: str,
    config: Dict[str, Any],
):
    import torch

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lm_head = teacher_model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Teacher does not expose an LM/output head.")

    torch.save(
        lm_head.state_dict(),
        output_path / "lm_head.pt",
    )

    tokenizer.save_pretrained(output_path)

    with (output_path / "doge_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def train_doge(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DOGe training requires a CUDA GPU in this benchmark."
        )

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    proxy_tokenizer = AutoTokenizer.from_pretrained(
        args.proxy_model,
        trust_remote_code=True,
    )

    if tokenizer.get_vocab() != proxy_tokenizer.get_vocab():
        raise ValueError(
            "DOGe v1 requires teacher and proxy to use the exact "
            "same token-to-id vocabulary mapping."
        )

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)

    proxy = AutoModelForCausalLM.from_pretrained(
        args.proxy_model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)

    if teacher.config.vocab_size != proxy.config.vocab_size:
        raise ValueError(
            "DOGe v1 expects teacher and proxy to share a tokenizer/vocabulary. "
            f"teacher vocab={teacher.config.vocab_size}, "
            f"proxy vocab={proxy.config.vocab_size}"
        )

    freeze_for_doge(teacher, proxy)

    proxy.eval()
    teacher.train()

    trainable = [
        p for p in teacher.parameters() if p.requires_grad
    ]

    print(
        f"[INFO] DOGe trainable parameters: "
        f"{sum(p.numel() for p in trainable):,}"
    )

    dataset = DOGeTrainingDataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(
            x, tokenizer.pad_token_id
        ),
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    optimizer.zero_grad()

    completed_steps = 0
    micro_steps = 0
    start_time = time.time()

    stop_training = False

    for epoch in range(args.num_train_epochs):
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            loss, lm_loss, kd_loss = doge_loss(
                teacher_model=teacher,
                proxy_model=proxy,
                batch=batch,
                anti_kd_coef=args.anti_kd_coef,
                kd_temperature=args.kd_temperature,
            )

            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            micro_steps += 1

            if micro_steps % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                completed_steps += 1

                print(
                    f"[DOGe] step={completed_steps} "
                    f"loss={loss.item():.6f} "
                    f"lm_loss={lm_loss.item():.6f} "
                    f"kd_loss={kd_loss.item():.6f}"
                )

                if (
                    args.max_steps is not None
                    and args.max_steps > 0
                    and completed_steps >= args.max_steps
                ):
                    stop_training = True
                    break

        if stop_training:
            break

    config = {
        "defense": "doge",
        "type": "anti_distillation_generator",
        "teacher_model": args.teacher_model,
        "proxy_model": args.proxy_model,
        "train_file": str(Path(args.train_file).resolve()),
        "anti_kd_coef": args.anti_kd_coef,
        "kd_temperature": args.kd_temperature,
        "train_lm_head_only": True,
        "reasoning_aware_mask": False,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "completed_steps": completed_steps,
        "max_length": args.max_length,
        "training_seconds": time.time() - start_time,
    }

    save_doge_checkpoint(
        teacher,
        tokenizer,
        args.output_dir,
        config,
    )

    print(f"[OK] Saved DOGe teacher to {args.output_dir}")


def load_defended_teacher(checkpoint: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_path = Path(checkpoint)

    config_path = checkpoint_path / "doge_config.json"
    lm_head_path = checkpoint_path / "lm_head.pt"

    if not config_path.exists():
        raise FileNotFoundError(
            f"DOGe config not found: {config_path}"
        )

    if not lm_head_path.exists():
        raise FileNotFoundError(
            f"DOGe LM-head checkpoint not found: {lm_head_path}"
        )

    doge_config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    base_teacher = doge_config["teacher_model"]

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_teacher,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto",
    )

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Teacher does not expose an LM/output head.")

    lm_head_state = torch.load(
        lm_head_path,
        map_location="cpu",
        weights_only=True,
    )
    lm_head.load_state_dict(lm_head_state)

    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"### User:\n{prompt}\n\n### Assistant:\n"


def generate_doge(args):
    import torch

    generation_start = time.time()

    model, tokenizer = load_defended_teacher(args.checkpoint)

    config_path = Path(args.checkpoint) / "doge_config.json"
    if config_path.exists():
        doge_config = json.loads(config_path.read_text())
    else:
        doge_config = {}

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    outputs = []

    for i, record in enumerate(read_jsonl(args.input), start=1):
        prompt = _pick(record, "prompt", "query", "question", "instruction")

        if prompt is None:
            raise KeyError(
                "DOGe input requires either 'prompt' or 'query'."
            )
        prompt = str(prompt)

        full_prompt = build_prompt(tokenizer, prompt)
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
        ).to(model.device)

        do_sample = args.temperature > 0

        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = args.temperature
            generated_ids = model.generate(**inputs, **generation_kwargs)

        new_tokens = generated_ids[
            0, inputs["input_ids"].shape[-1]:
        ]

        response = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        query_id = record.get(
            "query_id",
            record.get("id", f"doge_{i:06d}")
        )

        #
        # Keep BOTH the benchmark-native fields and
        # the DOGe protocol fields.
        #
        outputs.append({
            "id": record.get("id", query_id),
            "query_id": query_id,
            "dataset": record.get("dataset"),
            "method": record.get("method"),
            "prompt": prompt,
            "query": prompt,
            "teacher_response": response,
            "response": response,
            "ground_truth": record.get("ground_truth"),
            "teacher_model": doge_config.get(
                "teacher_model",
                args.checkpoint,
            ),
            "defense": "doge",
            "defense_role": DEFENSE_ROLE["doge"],
            "doge_teacher_checkpoint": args.checkpoint,
            "doge_config": doge_config,
            "metadata": {
                **record.get("metadata", {}),
                "defense": "doge",
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "generated_at": int(time.time()),
            },
        })

        if i % 10 == 0:
            print(f"[INFO] DOGe generated {i} responses")

    write_jsonl(outputs, output_path)

    generation_seconds = time.time() - generation_start

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else Path(args.output).parent / "defense_manifest.json"
    )
    ensure_dir(manifest_path.parent)
    if manifest_path.parent.name == "transcripts":
        run_dir = manifest_path.parent.parent
    elif output_path.parent.name == "transcripts":
        run_dir = output_path.parent.parent
    else:
        run_dir = manifest_path.parent

    config = {
        **doge_config,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    artifacts = {
        "lm_head_checkpoint": str((Path(args.checkpoint) / "lm_head.pt").resolve()),
        "doge_config": str((Path(args.checkpoint) / "doge_config.json").resolve()),
        "defended_transcript": str(output_path.resolve()),
    }
    cost = build_cost(
        wall_seconds=generation_seconds,
        num_queries=len(outputs),
        models=[doge_config.get("teacher_model", args.checkpoint)],
        extra_models=[doge_config["proxy_model"]] if doge_config.get("proxy_model") else None,
        data_files=[str(Path(args.input).resolve())],
        artifacts=artifacts,
        stage="defense_install",
        method="doge",
        extra={
            "defensive_training_seconds": doge_config.get("training_seconds"),
            "defensive_training_steps": doge_config.get("completed_steps"),
            "generation_seconds": generation_seconds,
            "num_generation_queries": len(outputs),
            "doge_teacher_checkpoint": str(Path(args.checkpoint).resolve()),
        },
    )
    manifest = DefenseManifest(
        defense="doge",
        type=DEFENSE_TYPE["doge"],
        teacher_model=doge_config.get("teacher_model", args.checkpoint),
        query_pool=str(Path(args.input).resolve()),
        output_transcript=str(output_path.resolve()),
        num_queries=len(outputs),
        config=config,
        artifacts=artifacts,
        cost=cost,
        run_id=run_dir.name,
    )
    manifest.save(manifest_path)

    print(
        f"[OK] Wrote {len(outputs)} defended responses "
        f"to {args.output}"
    )
    print(f"[OK] Wrote DOGe manifest to {manifest_path}")


def run_doge(args):
    run_id = args.run_id or _run_id()
    out_dir = Path(args.output_dir)
    if args.run_id and out_dir.name != run_id:
        out_dir = out_dir / run_id
    checkpoint_dir = out_dir / "checkpoints" / "doge_teacher"
    transcript_path = out_dir / "transcripts" / "defended_teacher.jsonl"
    manifest_path = out_dir / "defense_manifest.json"
    ensure_dir(out_dir / "artifacts")

    train_args = argparse.Namespace(
        teacher_model=args.teacher_model,
        proxy_model=args.proxy_model,
        train_file=args.train_file,
        output_dir=str(checkpoint_dir),
        anti_kd_coef=args.anti_kd_coef,
        kd_temperature=args.kd_temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
    )
    train_doge(train_args)

    generate_args = argparse.Namespace(
        checkpoint=str(checkpoint_dir),
        input=args.query_pool_path,
        output=str(transcript_path),
        manifest=str(manifest_path),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    generate_doge(generate_args)


def build_parser():
    parser = argparse.ArgumentParser(
        description="DOGe anti-distillation defense."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train = subparsers.add_parser("train")

    train.add_argument("--teacher-model", required=True)
    train.add_argument("--proxy-model", required=True)
    train.add_argument("--train-file", required=True)
    train.add_argument("--output-dir", required=True)

    train.add_argument("--anti-kd-coef", type=float, default=3e-5)
    train.add_argument("--kd-temperature", type=float, default=2.0)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
    )
    train.add_argument("--num-train-epochs", type=int, default=2)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--max-length", type=int, default=2048)

    generate = subparsers.add_parser("generate")

    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--input", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument(
        "--manifest",
        default=None,
        help="Optional path for defense_manifest.json.",
    )
    generate.add_argument("--max-new-tokens", type=int, default=256)
    generate.add_argument("--temperature", type=float, default=0.0)

    run = subparsers.add_parser("run")
    run.add_argument("--teacher_model", required=True)
    run.add_argument("--proxy_model", required=True)
    run.add_argument("--train_file", required=True)
    run.add_argument("--query_pool_path", required=True)
    run.add_argument("--output_dir", required=True)
    run.add_argument("--run_id", default=None)
    run.add_argument("--anti_kd_coef", type=float, default=3e-5)
    run.add_argument("--kd_temperature", type=float, default=2.0)
    run.add_argument("--learning_rate", type=float, default=5e-5)
    run.add_argument("--weight_decay", type=float, default=0.01)
    run.add_argument("--batch_size", type=int, default=1)
    run.add_argument("--gradient_accumulation_steps", type=int, default=16)
    run.add_argument("--num_train_epochs", type=int, default=2)
    run.add_argument("--max_steps", type=int, default=None)
    run.add_argument("--max_length", type=int, default=2048)
    run.add_argument("--max_new_tokens", type=int, default=256)
    run.add_argument("--temperature", type=float, default=0.0)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_doge(args)
    elif args.command == "generate":
        generate_doge(args)
    elif args.command == "run":
        run_doge(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
