import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from defenses.core.io_utils import read_jsonl, write_jsonl


# ============================================================
# Fingerprint hash / green-list construction
# ============================================================

class ADFPHash:
    """
    Deterministic red/green vocabulary partition.

    v1 implementation:
      seed = SHA256(secret_key || last-w token ids)
      green list = deterministic gamma fraction of vocabulary

    The exact hash configuration is saved as an artifact so the
    detector can reproduce exactly the same green lists.
    """

    def __init__(
        self,
        vocab_size: int,
        secret_key: str,
        gamma: float = 0.5,
        window_size: int = 2,
    ):
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.vocab_size = int(vocab_size)
        self.secret_key = str(secret_key)
        self.gamma = float(gamma)
        self.window_size = int(window_size)

        self.green_size = max(
            1,
            int(round(self.gamma * self.vocab_size)),
        )

    def context_seed(self, token_ids: List[int]) -> int:
        context = token_ids[-self.window_size:]

        payload = (
            self.secret_key
            + "|"
            + ",".join(str(int(x)) for x in context)
        ).encode("utf-8")

        digest = hashlib.sha256(payload).digest()

        # Keep seed inside signed 63-bit range accepted by torch.
        return int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        ) % (2**63 - 1)

    def green_indices(self, token_ids: List[int]) -> torch.Tensor:
        seed = self.context_seed(token_ids)

        # CPU RNG makes the artifact reproducible independent of GPU.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        permutation = torch.randperm(
            self.vocab_size,
            generator=generator,
            device="cpu",
        )

        return permutation[:self.green_size]

    def green_mask(
        self,
        token_ids: List[int],
        device,
        dtype=torch.float32,
    ) -> torch.Tensor:
        green = self.green_indices(token_ids)

        mask = torch.zeros(
            self.vocab_size,
            dtype=dtype,
            device=device,
        )

        mask[green.to(device)] = 1.0
        return mask

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash_scheme": "sha256_context_seed_v1",
            "secret_key": self.secret_key,
            "gamma": self.gamma,
            "window_size": self.window_size,
            "vocab_size": self.vocab_size,
            "green_size": self.green_size,
        }


# ============================================================
# ADFP antidistillation logits processor
# ============================================================

class ADFPLogitsProcessor(LogitsProcessor):
    """
    Implements the ADFP perturbation:

        Delta_t = q_t * (I[t in green] - L)

    where:
        q = proxy next-token distribution
        L = total proxy probability mass on green tokens

    Perturbed teacher scores:

        teacher_scores + lambda * Delta
    """

    def __init__(
        self,
        proxy_model,
        fingerprint_hash: ADFPHash,
        strength: float,
        *,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ):
        self.proxy_model = proxy_model
        self.fingerprint_hash = fingerprint_hash
        self.strength = float(strength)
        self.pad_token_id = int(pad_token_id) if pad_token_id is not None else None
        self.eos_token_id = int(eos_token_id) if eos_token_id is not None else None

        self.proxy_model.eval()

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        if scores.shape[-1] != self.fingerprint_hash.vocab_size:
            raise RuntimeError(
                "Teacher score vocabulary does not match "
                "ADFP fingerprint vocabulary."
            )

        proxy_device = next(
            self.proxy_model.parameters()
        ).device

        proxy_inputs = input_ids.to(proxy_device)
        attention_mask = None
        if self.pad_token_id is not None:
            attention_mask = proxy_inputs.ne(self.pad_token_id).long()

        proxy_outputs = self.proxy_model(
            input_ids=proxy_inputs,
            attention_mask=attention_mask,
            use_cache=False,
        )

        proxy_logits = proxy_outputs.logits[:, -1, :].float()

        if proxy_logits.shape[-1] != scores.shape[-1]:
            raise RuntimeError(
                "ADFP v1 requires teacher and proxy to use "
                "the same vocabulary size."
            )

        proxy_probs = torch.softmax(proxy_logits, dim=-1)

        output_scores = scores.float().clone()

        for batch_idx in range(input_ids.shape[0]):
            if self.eos_token_id is not None and self.eos_token_id < scores.shape[-1]:
                top_token = int(torch.argmax(scores[batch_idx]).item())
                if top_token == self.eos_token_id:
                    continue

            context_ids = input_ids[
                batch_idx
            ].detach().cpu().tolist()
            if self.pad_token_id is not None:
                context_ids = [tid for tid in context_ids if tid != self.pad_token_id]

            green_mask = self.fingerprint_hash.green_mask(
                context_ids,
                device=proxy_probs.device,
                dtype=proxy_probs.dtype,
            )

            q = proxy_probs[batch_idx]

            # L = proxy probability mass on green tokens.
            green_mass = torch.sum(q * green_mask)

            delta = q * (green_mask - green_mass)

            output_scores[batch_idx] = (
                output_scores[batch_idx]
                + self.strength * delta.to(
                    output_scores.device
                )
            )

        return output_scores.to(scores.dtype)


# ============================================================
# Model / prompt helpers
# ============================================================

def load_models(
    teacher_model_name: str,
    proxy_model_name: str,
):
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else None
    )

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name,
        trust_remote_code=True,
    )

    proxy_tokenizer = AutoTokenizer.from_pretrained(
        proxy_model_name,
        trust_remote_code=True,
    )

    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = (
            teacher_tokenizer.eos_token
        )
    if proxy_tokenizer.pad_token is None:
        proxy_tokenizer.pad_token = (
            proxy_tokenizer.eos_token
        )
    teacher_tokenizer.padding_side = "left"
    proxy_tokenizer.padding_side = "left"

    # v1 directly feeds teacher token ids into proxy.
    # Therefore require exact token-to-id compatibility.
    if (
        teacher_tokenizer.get_vocab()
        != proxy_tokenizer.get_vocab()
    ):
        raise ValueError(
            "ADFP v1 requires teacher and proxy to share "
            "the exact token-to-id vocabulary mapping."
        )

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )

    proxy = AutoModelForCausalLM.from_pretrained(
        proxy_model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )

    teacher.eval()
    proxy.eval()

    return teacher, proxy, teacher_tokenizer


def _batched(seq: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def _query_id(record: Dict[str, Any], fallback: str) -> str:
    return str(record.get("query_id", record.get("id", fallback)))


def _checkpoint_outputs(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    write_jsonl(records, tmp_path)
    tmp_path.replace(output_path)


def build_prompt(tokenizer, query: str) -> str:
    messages = [
        {
            "role": "user",
            "content": query,
        }
    ]

    if (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template
    ):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        f"### User:\n{query}\n\n"
        f"### Assistant:\n"
    )


def artifact_id(config: Dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:16]


# ============================================================
# ADFP generator
# ============================================================

def generate_adfp(args):
    start_time = time.time()

    teacher, proxy, tokenizer = load_models(
        args.teacher_model,
        args.proxy_model,
    )

    vocab_size = int(
        getattr(
            getattr(teacher, "config", None),
            "vocab_size",
            len(tokenizer.get_vocab()),
        )
    )

    fingerprint_hash = ADFPHash(
        vocab_size=vocab_size,
        secret_key=args.secret_key,
        gamma=args.gamma,
        window_size=args.window_size,
    )
    batch_size = int(getattr(args, "batch_size", 1) or 1)

    adfp_config = {
        "defense": "adfp",
        "type": "output_fingerprint",
        "teacher_model": args.teacher_model,
        "proxy_model": args.proxy_model,
        "strength_lambda": args.strength,
        "gamma": args.gamma,
        "window_size": args.window_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": batch_size,
        "hash_scheme": "sha256_context_seed_v1",
    }

    fp_artifact_id = artifact_id({
        **adfp_config,
        "secret_key": args.secret_key,
    })

    output_path = Path(args.output)
    run_dir = output_path.parent.parent

    artifact_dir = (
        Path(args.artifact_dir)
        if args.artifact_dir
        else run_dir / "artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    hash_config_path = artifact_dir / "hash_config.json"
    fingerprint_config_path = (
        artifact_dir / "fingerprint_config.json"
    )

    hash_config = fingerprint_hash.to_dict()
    hash_config["fingerprint_artifact_id"] = (
        fp_artifact_id
    )

    with hash_config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            hash_config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    fingerprint_config = {
        **adfp_config,
        "fingerprint_artifact_id": fp_artifact_id,
        "hash_config": str(hash_config_path),
    }

    with fingerprint_config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            fingerprint_config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logits_processor = LogitsProcessorList([
        ADFPLogitsProcessor(
            proxy_model=proxy,
            fingerprint_hash=fingerprint_hash,
            strength=args.strength,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    ])

    outputs = read_jsonl(output_path) if output_path.exists() else []
    existing_query_ids = {
        str(record.get("query_id", record.get("id", "")))
        for record in outputs
        if record.get("query_id") is not None or record.get("id") is not None
    }
    rows = [
        record
        for index, record in enumerate(read_jsonl(args.input), start=1)
        if _query_id(record, f"adfp_{index:06d}") not in existing_query_ids
    ]
    do_sample = args.temperature > 0

    if outputs:
        print(
            f"[INFO] ADFP resume: loaded {len(outputs)} existing responses "
            f"from {output_path}",
            flush=True,
        )

    total_batches = math.ceil(len(rows) / batch_size) if rows else 0
    for batch_index, batch in enumerate(_batched(rows, batch_size), start=1):
        batch_records = []
        prompts = []
        for offset, record in enumerate(batch, start=len(outputs) + 1):
            query = record.get(
                "query",
                record.get("prompt"),
            )

            if query is None:
                raise KeyError(
                    "ADFP input requires 'query' or 'prompt'."
                )

            query_id = _query_id(record, f"adfp_{offset:06d}")
            batch_records.append((record, query_id, query))
            prompts.append(build_prompt(tokenizer, query))

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(teacher.device)

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "logits_processor": logits_processor,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if do_sample:
            generation_kwargs.update({
                "temperature": args.temperature,
                "top_p": args.top_p,
            })

        print(
            f"[INFO] ADFP batch {batch_index}/{total_batches} "
            f"start: size={len(prompts)}, generated={len(outputs)}",
            flush=True,
        )

        with torch.no_grad():
            generated_ids = teacher.generate(
                **inputs,
                **generation_kwargs,
            )

        prompt_window = inputs["input_ids"].shape[-1]
        responses = tokenizer.batch_decode(
            generated_ids[:, prompt_window:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for record, response in zip(batch_records, responses):
            source_record, source_query_id, source_query = record
            response = response.strip()
            outputs.append({
                # Benchmark-native compatibility
                "id": source_record.get("id", source_query_id),
                "dataset": source_record.get("dataset"),
                "method": source_record.get("method"),
                "prompt": source_query,
                "teacher_response": response,
                "ground_truth": source_record.get(
                    "ground_truth"
                ),

                # ADFP protocol
                "query_id": source_query_id,
                "query": source_query,
                "response": response,
                "defense": "adfp",
                "defense_role":
                    "output_fingerprint_generator",
                "teacher_model": args.teacher_model,
                "proxy_model": args.proxy_model,
                "adfp_config": adfp_config,
                "fingerprint_artifact_id":
                    fp_artifact_id,

                "metadata": {
                    **source_record.get("metadata", {}),
                    "fingerprint_artifact_id":
                        fp_artifact_id,
                    "generated_at": int(time.time()),
                },
            })

        print(
            f"[INFO] ADFP batch {batch_index}/{total_batches} "
            f"done: generated={len(outputs)} responses",
            flush=True,
        )
        _checkpoint_outputs(outputs, output_path)

    _checkpoint_outputs(outputs, output_path)

    elapsed = time.time() - start_time

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else run_dir / "defense_manifest.json"
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "defense": "adfp",
        "type": "output_fingerprint",
        "teacher_model": args.teacher_model,
        "proxy_model": args.proxy_model,
        "query_pool": args.input,
        "output_transcript": args.output,
        "num_queries": len(outputs),
        "adfp_config": adfp_config,
        "fingerprint_artifact_id":
            fp_artifact_id,
        "artifacts": {
            "hash_config":
                str(hash_config_path),
            "fingerprint_config":
                str(fingerprint_config_path),
        },
        "cost": {
            "num_teacher_generations":
                len(outputs),
            "num_proxy_assisted_generations":
                len(outputs),
            "generation_seconds": elapsed,
        },
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[OK] Wrote {len(outputs)} "
        f"ADFP responses to {args.output}"
    )
    print(
        f"[OK] Fingerprint artifact: "
        f"{fp_artifact_id}"
    )
    print(
        f"[OK] Manifest: {manifest_path}"
    )



# ============================================================
# ADFP detector
# ============================================================

def load_student_model(
    student_model: str,
    student_base_model: Optional[str] = None,
):
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else None
    )

    tokenizer_source = student_model

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
        )
    except Exception:
        if student_base_model is None:
            raise

        tokenizer = AutoTokenizer.from_pretrained(
            student_base_model,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if student_base_model:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "PEFT is required when --student-base-model "
                "is used."
            ) from exc

        base = AutoModelForCausalLM.from_pretrained(
            student_base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(
            base,
            student_model,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            student_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )

    model.eval()

    return model, tokenizer


def load_fingerprint_artifacts(
    fingerprint_config_path: str,
):
    path = Path(fingerprint_config_path)

    config = json.loads(
        path.read_text(encoding="utf-8")
    )

    hash_path = Path(config["hash_config"])

    if not hash_path.is_absolute():
        candidate = path.parent / hash_path.name
        if candidate.exists():
            hash_path = candidate

    hash_config = json.loads(
        hash_path.read_text(encoding="utf-8")
    )

    return config, hash_config, hash_path


def build_detection_contexts(
    transcript_path: str,
    tokenizer,
    window_size: int,
    max_contexts: Optional[int] = None,
    min_context_tokens: int = 4,
):
    """
    Construct multiple prefix contexts from fingerprint traces.

    v1 assumes the detector student uses the same token-id
    vocabulary as the fingerprint teacher.
    """

    contexts = []
    seen_suffixes = set()

    for record in read_jsonl(transcript_path):
        query = record.get(
            "query",
            record.get("prompt"),
        )

        response = record.get(
            "response",
            record.get("teacher_response"),
        )

        if query is None or response is None:
            continue

        prompt_text = build_prompt(
            tokenizer,
            query,
        )

        full_text = prompt_text + response

        token_ids = tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        if len(token_ids) <= window_size:
            continue

        start = max(
            min_context_tokens,
            window_size,
        )

        # Each prefix becomes one next-token detection context.
        for end in range(start, len(token_ids)):
            context_ids = token_ids[:end]

            suffix = tuple(
                context_ids[-window_size:]
            )

            # Required for the independence assumption used
            # by the fingerprint statistical test.
            if suffix in seen_suffixes:
                continue

            seen_suffixes.add(suffix)

            contexts.append({
                "query_id": record.get(
                    "query_id",
                    record.get("id"),
                ),
                "input_ids": context_ids,
                "suffix": list(suffix),
            })

            if (
                max_contexts is not None
                and len(contexts) >= max_contexts
            ):
                return contexts

    return contexts


@torch.no_grad()
def compute_gtp(
    model,
    contexts,
    fingerprint_hash: ADFPHash,
):
    """
    Open-weight ADFP detection.

    GTP = average student next-token probability mass
          assigned to the green list.
    """

    if not contexts:
        raise ValueError(
            "No valid ADFP detection contexts were built."
        )

    device = next(model.parameters()).device

    green_probabilities = []

    for i, ctx in enumerate(contexts, start=1):
        input_ids = torch.tensor(
            [ctx["input_ids"]],
            dtype=torch.long,
            device=device,
        )

        outputs = model(
            input_ids=input_ids,
            use_cache=False,
        )

        logits = outputs.logits[
            0, -1, :
        ].float()

        if logits.shape[-1] != fingerprint_hash.vocab_size:
            raise RuntimeError(
                "Student vocabulary size does not match "
                "the ADFP fingerprint artifact."
            )

        probs = torch.softmax(
            logits,
            dim=-1,
        )

        green_indices = (
            fingerprint_hash.green_indices(
                ctx["input_ids"]
            ).to(probs.device)
        )

        green_prob = probs[
            green_indices
        ].sum().item()

        green_probabilities.append(
            green_prob
        )

        if i % 100 == 0:
            print(
                f"[INFO] Evaluated "
                f"{i}/{len(contexts)} "
                f"fingerprint contexts"
            )

    gtp = (
        sum(green_probabilities)
        / len(green_probabilities)
    )

    return gtp, green_probabilities


def hoeffding_fingerprint_score(
    gtp: float,
    gamma: float,
    n: int,
):
    """
    Conservative ADFP fingerprint p-value:

        p = exp(-2 n (g_obs - gamma)^2)

    when g_obs > gamma.

    Benchmark score is -log10(p), so higher means
    stronger fingerprint evidence.
    """

    excess = max(
        float(gtp) - float(gamma),
        0.0,
    )

    log_p = -2.0 * n * excess * excess

    if excess <= 0:
        p_value = 1.0
        score = 0.0
    else:
        # Avoid numerical underflow in exp().
        p_value = (
            math.exp(log_p)
            if log_p > -745
            else 0.0
        )

        score = (
            -log_p / math.log(10.0)
        )

    return {
        "p_value": p_value,
        "log_p_value": log_p,
        "neg_log10_p": score,
        "gtp": float(gtp),
        "gamma": float(gamma),
        "num_contexts": int(n),
    }


def detect_adfp(args):
    start_time = time.time()

    fingerprint_config, hash_config, hash_path = (
        load_fingerprint_artifacts(
            args.fingerprint_config
        )
    )

    teacher_model_name = (
        fingerprint_config["teacher_model"]
    )

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name,
        trust_remote_code=True,
    )

    student, student_tokenizer = (
        load_student_model(
            args.student_model,
            args.student_base_model,
        )
    )

    #
    # v1 checkpoint detector uses exact tokenizer mapping.
    #
    if (
        teacher_tokenizer.get_vocab()
        != student_tokenizer.get_vocab()
    ):
        raise ValueError(
            "ADFP detector v1 requires student and "
            "fingerprint teacher to use the exact same "
            "token-to-id vocabulary mapping. "
            "Cross-tokenizer detection is a future extension."
        )

    fingerprint_hash = ADFPHash(
        vocab_size=hash_config["vocab_size"],
        secret_key=hash_config["secret_key"],
        gamma=hash_config["gamma"],
        window_size=hash_config["window_size"],
    )

    contexts = build_detection_contexts(
        transcript_path=args.transcript,
        tokenizer=student_tokenizer,
        window_size=fingerprint_hash.window_size,
        max_contexts=args.max_contexts,
        min_context_tokens=args.min_context_tokens,
    )

    print(
        f"[INFO] Built {len(contexts)} "
        f"unique ADFP detection contexts"
    )

    gtp, per_context = compute_gtp(
        student,
        contexts,
        fingerprint_hash,
    )

    stats = hoeffding_fingerprint_score(
        gtp=gtp,
        gamma=fingerprint_hash.gamma,
        n=len(contexts),
    )

    elapsed = time.time() - start_time

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_path = (
        output_dir / "detector_rows.jsonl"
    )
    report_path = (
        output_dir / "detector_report.json"
    )
    manifest_path = (
        output_dir / "detector_manifest.json"
    )

    model_id = (
        args.model_id
        if args.model_id
        else args.student_model
    )

    row = {
        "id": (
            f"adfp_detection::{model_id}"
        ),
        "model_id": model_id,
        "label": args.label,
        "score": stats["neg_log10_p"],
        "score_direction": "higher",
        "score_name": "negative_log10_p_value",
        "detector": "adfp",
        "defense_run_id":
            args.defense_run_id,
        "attack_run_id":
            args.attack_run_id,
        "gtp": stats["gtp"],
        "gamma": stats["gamma"],
        "p_value": stats["p_value"],
        "log_p_value": stats["log_p_value"],
        "num_contexts":
            stats["num_contexts"],
    }

    write_jsonl(
        [row],
        str(rows_path),
    )

    report = {
        "detector": "adfp",
        "type": "checkpoint_detector",
        "model_id": model_id,
        "student_checkpoint":
            args.student_model,
        "student_base_model":
            args.student_base_model,
        "label": args.label,
        "score":
            stats["neg_log10_p"],
        "score_name":
            "negative_log10_p_value",
        "score_direction": "higher",
        "gtp": stats["gtp"],
        "gamma": stats["gamma"],
        "p_value": stats["p_value"],
        "log_p_value":
            stats["log_p_value"],
        "num_contexts":
            stats["num_contexts"],
        "same_tokenizer_only": True,
        "fingerprint_artifact_id":
            fingerprint_config.get(
                "fingerprint_artifact_id"
            ),
        "fingerprint_config":
            args.fingerprint_config,
        "hash_config":
            str(hash_path),
        "transcript":
            args.transcript,
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    manifest = {
        "detector": "adfp",
        "type": "checkpoint_detector",
        "defense_run_id":
            args.defense_run_id,
        "attack_run_id":
            args.attack_run_id,
        "student_checkpoint":
            args.student_model,
        "student_base_model":
            args.student_base_model,
        "fingerprint_artifacts": {
            "fingerprint_config":
                args.fingerprint_config,
            "hash_config":
                str(hash_path),
            "fingerprint_artifact_id":
                fingerprint_config.get(
                    "fingerprint_artifact_id"
                ),
        },
        "input_transcript":
            args.transcript,
        "output_rows":
            str(rows_path),
        "output_report":
            str(report_path),
        "detector_config": {
            "max_contexts":
                args.max_contexts,
            "min_context_tokens":
                args.min_context_tokens,
            "same_tokenizer_only": True,
        },
        "cost": {
            "num_student_forward_contexts":
                len(contexts),
            "detector_seconds":
                elapsed,
        },
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[OK] ADFP detector complete"
    )
    print(
        f"GTP: {stats['gtp']:.6f}"
    )
    print(
        f"p-value: {stats['p_value']}"
    )
    print(
        f"-log10(p): "
        f"{stats['neg_log10_p']:.6f}"
    )
    print(
        f"Report: {report_path}"
    )
    print(
        f"Manifest: {manifest_path}"
    )


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "ADFP output fingerprint defense."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    generate = subparsers.add_parser(
        "generate"
    )

    generate.add_argument(
        "--teacher-model",
        required=True,
    )
    generate.add_argument(
        "--proxy-model",
        required=True,
    )
    generate.add_argument(
        "--input",
        required=True,
    )
    generate.add_argument(
        "--output",
        required=True,
    )
    generate.add_argument(
        "--manifest",
        default=None,
    )
    generate.add_argument(
        "--artifact-dir",
        default=None,
    )

    generate.add_argument(
        "--secret-key",
        default="adfp-benchmark-key-v1",
    )
    generate.add_argument(
        "--gamma",
        type=float,
        default=0.5,
    )
    generate.add_argument(
        "--window-size",
        type=int,
        default=2,
    )
    generate.add_argument(
        "--strength",
        type=float,
        default=140.0,
    )

    generate.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    generate.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )
    generate.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    generate.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    detect = subparsers.add_parser(
        "detect"
    )

    detect.add_argument(
        "--student-model",
        required=True,
        help=(
            "Student checkpoint/model path. "
            "For a PEFT adapter, also provide "
            "--student-base-model."
        ),
    )
    detect.add_argument(
        "--student-base-model",
        default=None,
    )
    detect.add_argument(
        "--transcript",
        required=True,
    )
    detect.add_argument(
        "--fingerprint-config",
        required=True,
    )
    detect.add_argument(
        "--output-dir",
        required=True,
    )

    detect.add_argument(
        "--model-id",
        default=None,
    )
    detect.add_argument(
        "--label",
        default="unknown",
    )
    detect.add_argument(
        "--defense-run-id",
        default=None,
    )
    detect.add_argument(
        "--attack-run-id",
        default=None,
    )
    detect.add_argument(
        "--max-contexts",
        type=int,
        default=1000,
    )
    detect.add_argument(
        "--min-context-tokens",
        type=int,
        default=4,
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        generate_adfp(args)
    elif args.command == "detect":
        detect_adfp(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
