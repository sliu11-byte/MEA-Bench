from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import http.client
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, set_seed

from .seqkd_train import _ensure_tokenizer_chat_defaults, _prompt_prefix_text
from .stage1_transcript import (
    DecodeConfig,
    PromptOrdering,
    QueryAttempt,
    QueryLedger,
    TranscriptBundle,
    TranscriptBudgetError,
    TranscriptConfigMismatchError,
    TranscriptManifest,
    TranscriptRecord,
    build_prompt_ordering,
    build_prompt_specs,
    git_commit,
    load_transcript_bundle,
    resolve_bundle_dir,
    save_transcript_bundle,
    stable_hash,
    transcript_hash_from_records,
)


ALLOWED_BUDGETS = {100, 1000, 10000}
SCHEMA_VERSION = "stage1_teacher_transcript_v1"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


@dataclass(frozen=True)
class TeacherTranscriptConfig:
    teacher_backend: str
    teacher_model_path: str
    query_pool_path: str
    query_ordering_path: str
    seed: int
    teacher_decode: Mapping[str, Any]
    output_root: str
    teacher_model_name: str | None = None
    teacher_base_url: str | None = None
    teacher_api_key: str | None = None
    teacher_mode: str = "completion"
    prompt_pool_id: str | None = None


class TeacherTranscriptError(RuntimeError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            default = match.group(2)
            resolved = os.environ.get(name)
            if resolved is not None:
                return resolved
            if default is not None:
                return default
            raise TeacherTranscriptError(f"missing environment variable: {name}")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TeacherTranscriptError(f"config file must contain a mapping: {path}")
    return dict(_expand_env(payload))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TeacherTranscriptError(f"json file must contain an object: {path}")
    return dict(payload)


def _config_hash(payload: Mapping[str, Any]) -> str:
    return stable_hash(payload)


def _normalize_decode_config(value: Mapping[str, Any]) -> dict[str, Any]:
    strategy = str(value.get("strategy", "greedy")).lower()
    settings = dict(value.get("settings", {}))
    if strategy not in {"greedy", "beam"}:
        raise TeacherTranscriptError(f"unsupported teacher_decode.strategy: {strategy!r}")
    if strategy == "beam" and int(settings.get("num_beams", 0)) != 5:
        raise TeacherTranscriptError("beam teacher decode must use num_beams=5.")
    if strategy == "greedy":
        settings.setdefault("num_beams", 1)
    return {"strategy": strategy, "settings": settings}


def _load_pool_records(path: Path) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    payload = _read_json(path)
    meta = dict(payload.get("meta", {}))
    records = payload.get("data", [])
    if not isinstance(records, list):
        raise TeacherTranscriptError("query pool data must be a list.")
    prompt_pool_id = str(meta.get("prompt_pool_id") or meta.get("tier_id") or "openllm6_master_pool")
    normalized_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise TeacherTranscriptError("each query pool record must be an object.")
        prompt_id = str(record.get("id") or record.get("prompt_id"))
        prompt_text = str(record.get("prompt") or record.get("prompt_text"))
        if not prompt_id or not prompt_text:
            raise TeacherTranscriptError("query pool records must contain id/prompt.")
        if prompt_id in seen_ids:
            raise TeacherTranscriptError(f"duplicate prompt_id in query pool: {prompt_id}")
        normalized_prompt = " ".join(prompt_text.casefold().split())
        if normalized_prompt in seen_prompts:
            raise TeacherTranscriptError(f"duplicate prompt text in query pool: {prompt_id}")
        seen_ids.add(prompt_id)
        seen_prompts.add(normalized_prompt)
        normalized_record = dict(record)
        normalized_record["prompt_id"] = prompt_id
        normalized_record["prompt_text"] = prompt_text
        normalized_records.append(normalized_record)
    prompt_pool_hash = stable_hash([(record["prompt_id"], record["prompt_text"]) for record in normalized_records])
    return prompt_pool_id, prompt_pool_hash, normalized_records, meta


def _load_ordering(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    ordered_prompt_ids = payload.get("ordered_prompt_ids", [])
    ordered_indices = payload.get("ordered_indices", [])
    if not isinstance(ordered_prompt_ids, list) or not isinstance(ordered_indices, list):
        raise TeacherTranscriptError("ordering file must contain ordered_prompt_ids and ordered_indices lists.")
    if len(ordered_prompt_ids) != len(ordered_indices):
        raise TeacherTranscriptError("ordering ids and indices must have the same length.")
    return {
        "prompt_pool_id": str(payload.get("prompt_pool_id", "")),
        "seed": int(payload.get("seed", 0)),
        "prompt_pool_hash": str(payload.get("prompt_pool_hash", "")),
        "ordering_hash": str(payload.get("ordering_hash", "")),
        "ordered_prompt_ids": [str(item) for item in ordered_prompt_ids],
        "ordered_indices": [int(item) for item in ordered_indices],
    }


def _prompt_specs_from_pool(records: list[dict[str, Any]]) -> list[Any]:
    return build_prompt_specs(
        [str(record["prompt_text"]) for record in records],
        prompt_ids=[str(record["prompt_id"]) for record in records],
    )


def _render_prompt(tokenizer: PreTrainedTokenizerBase, prompt_text: str) -> str:
    return _prompt_prefix_text(tokenizer, prompt_text)


def _decode_config_for_manifest(decode_config: Mapping[str, Any]) -> dict[str, Any]:
    return {"strategy": str(decode_config["strategy"]), "settings": dict(decode_config["settings"])}


def _generate_local_hf(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    decode_config: Mapping[str, Any],
    *,
    seed: int,
) -> str:
    rendered_prompt = _render_prompt(tokenizer, prompt_text)
    inputs = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    settings = dict(decode_config["settings"])
    strategy = str(decode_config["strategy"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generate_kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": int(settings.get("max_new_tokens", 64)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if strategy == "beam":
        generate_kwargs["num_beams"] = 5
        generate_kwargs["num_return_sequences"] = 1
    else:
        generate_kwargs["num_beams"] = 1
    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip()


def _generate_vllm_openai(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    prompt_text: str,
    decode_config: Mapping[str, Any],
    teacher_mode: str = "completion",
) -> str:
    strategy = str(decode_config["strategy"])
    settings = dict(decode_config["settings"])
    if strategy == "beam":
        raise TeacherTranscriptError("vllm_openai backend only supports greedy decoding in this runner.")

    mode = teacher_mode.lower().strip()
    if mode in {"chat", "chat.completions", "chat_completions"}:
        route = "chat/completions"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(settings.get("max_new_tokens", 64)),
            "n": 1,
        }
    elif mode in {"completion", "completions", "text"}:
        route = "completions"
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": int(settings.get("max_new_tokens", 64)),
            "n": 1,
        }
    else:
        raise TeacherTranscriptError(f"unsupported vllm_openai teacher_mode: {teacher_mode!r}")

    retry_count = int(os.environ.get("ATTACK_QUERY_RETRIES", "3"))
    retry_delay = float(os.environ.get("ATTACK_QUERY_RETRY_DELAY_SECONDS", "2"))
    if retry_count < 0 or retry_delay < 0:
        raise TeacherTranscriptError("ATTACK_QUERY_RETRIES and ATTACK_QUERY_RETRY_DELAY_SECONDS must be non-negative")
    transient_http_codes = {408, 429, 500, 502, 503, 504}
    request_url = f"{base_url.rstrip('/')}/{route}"
    for attempt in range(retry_count + 1):
        request = urllib.request.Request(
            request_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key or 'EMPTY'}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:  # pragma: no cover - network/runtime path
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            if exc.code in transient_http_codes and attempt < retry_count:
                time.sleep(retry_delay * (2**attempt))
                continue
            message = f"vLLM request failed after {attempt + 1} attempt(s): HTTP {exc.code} {exc.reason}"
            if detail:
                message += f"; response body: {detail[:2000]}"
            raise TeacherTranscriptError(message) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout, http.client.HTTPException) as exc:
            if attempt < retry_count:  # pragma: no cover - network/runtime path
                time.sleep(retry_delay * (2**attempt))
                continue
            raise TeacherTranscriptError(
                f"vLLM request failed after {attempt + 1} attempt(s): {type(exc).__name__}: {exc}"
            ) from exc
    choices = payload.get("choices", [])
    if not choices:
        raise TeacherTranscriptError("vLLM response did not include any choices.")
    choice = choices[0]
    if route == "chat/completions":
        text = (choice.get("message") or {}).get("content", "")
    else:
        text = choice.get("text", "")
    return str(text).strip()


def _load_config(config_path: Path, backend_override: str | None = None) -> TeacherTranscriptConfig:
    payload = _load_yaml(config_path)
    teacher_backend = str(backend_override or payload.get("teacher_backend", "local_hf"))
    teacher_model_path = str(payload.get("teacher_model_path", "")).strip()
    query_pool_path = str(payload.get("query_pool_path", "data/query_pool.json"))
    query_ordering_path = str(payload.get("query_ordering_path", "data/query_ordering.json"))
    teacher_decode = _normalize_decode_config(dict(payload.get("teacher_decode", {})))
    output_root = str(payload.get("output_root", "results/stage1"))
    seed = int(payload.get("seed", 0))
    if not teacher_model_path:
        raise TeacherTranscriptError("teacher_model_path is required.")
    return TeacherTranscriptConfig(
        teacher_backend=teacher_backend,
        teacher_model_path=teacher_model_path,
        query_pool_path=query_pool_path,
        query_ordering_path=query_ordering_path,
        seed=seed,
        teacher_decode=teacher_decode,
        output_root=output_root,
        teacher_model_name=payload.get("teacher_model_name"),
        teacher_base_url=payload.get("teacher_base_url"),
        teacher_api_key=payload.get("teacher_api_key"),
        teacher_mode=str(payload.get("teacher_mode", "completion")),
        prompt_pool_id=payload.get("prompt_pool_id"),
    )


def _bundle_identity(
    config: TeacherTranscriptConfig,
    prompt_pool_id: str,
    prompt_pool_hash: str,
    ordering_hash: str,
) -> dict[str, Any]:
    return {
        "teacher_backend": config.teacher_backend,
        "teacher_model_path": config.teacher_model_path,
        "teacher_model_name": config.teacher_model_name,
        "teacher_decode": config.teacher_decode,
        "teacher_mode": config.teacher_mode,
        "prompt_pool_id": prompt_pool_id,
        "prompt_pool_hash": prompt_pool_hash,
        "ordering_hash": ordering_hash,
        "seed": config.seed,
    }


def _sidecar_matches_identity(
    sidecar: Mapping[str, Any],
    identity: Mapping[str, Any],
    config_hash: str,
) -> bool:
    existing_hash = sidecar.get("config_hash")
    if existing_hash == config_hash:
        return True

    # Older sidecars included the ephemeral vLLM URL in config_hash. Validate
    # every other identity field against that legacy hash so a new node/port
    # can safely reuse the same semantic transcript.
    legacy_identity = dict(identity)
    legacy_identity["teacher_base_url"] = sidecar.get("teacher_base_url")
    return existing_hash == _config_hash(legacy_identity)


def _write_teacher_build_sidecar(
    bundle_dir: Path,
    *,
    config_hash: str,
    config_path: Path,
    query_pool_path: Path,
    query_ordering_path: Path,
    manifest: TranscriptManifest,
    ledger: QueryLedger,
    config: TeacherTranscriptConfig,
) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(),
        "created_at": _utc_now_iso(),
        "config_hash": config_hash,
        "config_path": str(config_path),
        "query_pool_path": str(query_pool_path),
        "query_ordering_path": str(query_ordering_path),
        "teacher_backend": config.teacher_backend,
        "teacher_model_path": config.teacher_model_path,
        "teacher_model_name": config.teacher_model_name,
        "teacher_base_url": config.teacher_base_url,
        "teacher_mode": config.teacher_mode,
        "seed": config.seed,
        "bundle_dir": str(bundle_dir),
        "manifest_path": manifest.transcript_path.replace("transcript.jsonl", "manifest.json"),
        "ledger_hash": stable_hash(ledger.to_dict()),
        "transcript_hash": manifest.transcript_hash,
        "ordering_hash": manifest.prompt_ordering_hash,
        "requested_budget": manifest.requested_budget,
        "actual_successful_queries": manifest.actual_successful_queries,
    }
    sidecar = bundle_dir / "teacher_build.json"
    sidecar.write_text(_canonical_json(payload), encoding="utf-8")
    return sidecar


def _load_existing_sidecar(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TeacherTranscriptError(f"invalid sidecar payload: {path}")
    return dict(payload)


def _build_records(
    *,
    ordered_prompts: list[dict[str, Any]],
    existing_records: list[TranscriptRecord],
    teacher_model_id: str,
    decode_config: Mapping[str, Any],
    seed: int,
    backend: str,
    model: Any | None,
    tokenizer: PreTrainedTokenizerBase | None,
    base_url: str | None,
    api_key: str | None,
    teacher_mode: str = "completion",
) -> list[TranscriptRecord]:
    records = list(existing_records)
    start_index = len(records)

    def generate(item: tuple[int, dict[str, Any]]) -> TranscriptRecord:
        order_index, spec = item
        prompt_id = str(spec["prompt_id"])
        prompt_text = str(spec["prompt_text"])
        query_id = f"{stable_hash({'prompt_id': prompt_id, 'order_index': order_index})[:12]}:{order_index:06d}"
        per_prompt_seed = seed + order_index
        if backend == "local_hf":
            if model is None or tokenizer is None:
                raise TeacherTranscriptError("local_hf backend requires a model and tokenizer.")
            teacher_response = _generate_local_hf(
                model,
                tokenizer,
                prompt_text,
                decode_config,
                seed=per_prompt_seed,
            )
        else:
            if base_url is None:
                raise TeacherTranscriptError("vllm_openai backend requires a base_url.")
            teacher_response = _generate_vllm_openai(
                base_url=base_url,
                api_key=api_key or "EMPTY",
                model_name=teacher_model_id,
                prompt_text=prompt_text,
                decode_config=decode_config,
                teacher_mode=teacher_mode,
            )
        return TranscriptRecord(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            teacher_response=teacher_response,
            teacher_model_id=teacher_model_id,
            decode_config=dict(decode_config),
            seed=seed,
            order_index=order_index,
            query_id=query_id,
            query_status="success",
            created_at=_utc_now_iso(),
        )

    pending = list(enumerate(ordered_prompts[start_index:], start=start_index))
    concurrency = int(os.environ.get("ATTACK_QUERY_CONCURRENCY", "8"))
    if concurrency < 1:
        raise TeacherTranscriptError("ATTACK_QUERY_CONCURRENCY must be positive")
    if backend == "local_hf" or concurrency == 1:
        records.extend(generate(item) for item in pending)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            records.extend(executor.map(generate, pending))
    return records


def build_teacher_transcript(
    *,
    config_path: Path,
    budget: int,
    backend_override: str | None = None,
    output_dir_override: Path | None = None,
    dry_run: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    if budget not in ALLOWED_BUDGETS:
        raise TeacherTranscriptError(f"budget must be one of {sorted(ALLOWED_BUDGETS)}.")

    config = _load_config(config_path, backend_override=backend_override)
    query_pool_path = Path(config.query_pool_path)
    query_ordering_path = Path(config.query_ordering_path)
    prompt_pool_id, prompt_pool_hash, pool_records, pool_meta = _load_pool_records(query_pool_path)
    ordering = _load_ordering(query_ordering_path)
    if config.prompt_pool_id:
        prompt_pool_id = str(config.prompt_pool_id)
    if ordering["prompt_pool_id"] and ordering["prompt_pool_id"] != prompt_pool_id:
        raise TranscriptConfigMismatchError(
            f"prompt_pool_id mismatch: pool={prompt_pool_id!r}, ordering={ordering['prompt_pool_id']!r}"
        )
    if ordering["prompt_pool_hash"] != prompt_pool_hash:
        raise TranscriptConfigMismatchError("query ordering does not match query pool hash.")
    if len(ordering["ordered_prompt_ids"]) != len(pool_records):
        raise TranscriptConfigMismatchError("ordering length does not match query pool length.")

    prompt_map = {record["prompt_id"]: record for record in pool_records}
    ordered_records = [prompt_map[prompt_id] for prompt_id in ordering["ordered_prompt_ids"]]
    if any(record["prompt_id"] != ordering["ordered_prompt_ids"][index] for index, record in enumerate(ordered_records)):
        raise TranscriptConfigMismatchError("ordering/pool prompt IDs disagree.")
    ordering_hash = str(ordering["ordering_hash"]) or stable_hash(
        {
            "prompt_pool_id": prompt_pool_id,
            "seed": ordering["seed"] or config.seed,
            "prompt_pool_hash": prompt_pool_hash,
            "ordered_prompt_ids": tuple(ordering["ordered_prompt_ids"]),
        }
    )
    ordering_obj = PromptOrdering(
        prompt_pool_id=prompt_pool_id,
        seed=int(ordering["seed"] or config.seed),
        prompt_pool_hash=prompt_pool_hash,
        ordering_hash=ordering_hash,
        ordered_prompt_ids=tuple(ordering["ordered_prompt_ids"]),
        ordered_indices=tuple(ordering["ordered_indices"]),
    )

    teacher_decode = _decode_config_for_manifest(config.teacher_decode)
    output_root = Path(output_dir_override or config.output_root)
    bundle_dir = resolve_bundle_dir(
        base_dir=output_root,
        prompt_pool_id=prompt_pool_id,
        prompt_pool_hash=prompt_pool_hash,
        prompt_ordering_hash=ordering_hash,
        teacher_model_id=config.teacher_model_path,
        decode_config=teacher_decode,
        seed=config.seed,
    )
    bundle_identity = _bundle_identity(config, prompt_pool_id, prompt_pool_hash, ordering_hash)
    config_hash = _config_hash(bundle_identity)
    sidecar_path = bundle_dir / "teacher_build.json"
    existing_sidecar = _load_existing_sidecar(sidecar_path)
    if existing_sidecar is not None and not _sidecar_matches_identity(
        existing_sidecar, bundle_identity, config_hash
    ):
        raise TeacherTranscriptError("existing transcript bundle cannot be reused with a different config hash.")

    bundle_exists = bundle_dir.exists()
    existing_bundle: TranscriptBundle | None = None
    if bundle_exists:
        existing_bundle = load_transcript_bundle(
            bundle_dir,
            expected_prompt_pool_id=prompt_pool_id,
            expected_prompt_pool_hash=prompt_pool_hash,
            expected_prompt_ordering_hash=ordering_hash,
            expected_teacher_model_id=config.teacher_model_path,
            expected_decode_config=teacher_decode,
            expected_seed=config.seed,
        )

    if existing_bundle is not None and len(existing_bundle.records) >= budget:
        if validate_only or dry_run:
            return {
                "status": "validated" if validate_only else "dry_run",
                "bundle_dir": str(bundle_dir),
                "records": len(existing_bundle.records),
                "budget": budget,
                "prompt_pool_id": prompt_pool_id,
                "prompt_pool_hash": prompt_pool_hash,
                "ordering_hash": ordering_hash,
                "config_hash": config_hash,
            }
        return {
            "status": "reused",
            "bundle_dir": str(bundle_dir),
            "records": len(existing_bundle.records),
            "budget": budget,
            "prompt_pool_id": prompt_pool_id,
            "prompt_pool_hash": prompt_pool_hash,
            "ordering_hash": ordering_hash,
            "config_hash": config_hash,
        }

    if validate_only:
        if existing_bundle is None:
            raise TeacherTranscriptError("validate-only requested but transcript bundle does not exist.")
        return {
            "status": "validated",
            "bundle_dir": str(bundle_dir),
            "records": len(existing_bundle.records),
            "budget": budget,
            "prompt_pool_id": prompt_pool_id,
            "prompt_pool_hash": prompt_pool_hash,
            "ordering_hash": ordering_hash,
            "config_hash": config_hash,
        }

    if dry_run:
        return {
            "status": "dry_run",
            "bundle_dir": str(bundle_dir),
            "records": 0 if existing_bundle is None else len(existing_bundle.records),
            "budget": budget,
            "prompt_pool_id": prompt_pool_id,
            "prompt_pool_hash": prompt_pool_hash,
            "ordering_hash": ordering_hash,
            "config_hash": config_hash,
            "will_generate": max(0, budget - (0 if existing_bundle is None else len(existing_bundle.records))),
        }

    set_seed(config.seed)
    torch.manual_seed(config.seed)

    existing_records = [] if existing_bundle is None else list(existing_bundle.records)
    if existing_records:
        existing_prompt_ids = [record.prompt_id for record in existing_records]
        expected_prefix = ordering["ordered_prompt_ids"][: len(existing_records)]
        if existing_prompt_ids != expected_prefix:
            raise TranscriptConfigMismatchError("existing transcript prefix does not match ordering prefix.")

    if config.teacher_backend == "local_hf":
        tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_path)
        _ensure_tokenizer_chat_defaults(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(config.teacher_model_path)
        model.eval()
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        teacher_model_id = config.teacher_model_name or config.teacher_model_path
        records = _build_records(
            ordered_prompts=ordered_records[:budget],
            existing_records=existing_records,
            teacher_model_id=teacher_model_id,
            decode_config=teacher_decode,
            seed=config.seed,
            backend="local_hf",
            model=model,
            tokenizer=tokenizer,
            base_url=None,
            api_key=None,
            teacher_mode=config.teacher_mode,
        )
    elif config.teacher_backend == "vllm_openai":
        base_url = config.teacher_base_url
        if not base_url:
            raise TeacherTranscriptError("vllm_openai backend requires teacher_base_url / STAGE1_TEACHER_BASE_URL.")
        teacher_model_id = config.teacher_model_name or config.teacher_model_path
        records = _build_records(
            ordered_prompts=ordered_records[:budget],
            existing_records=existing_records,
            teacher_model_id=teacher_model_id,
            decode_config=teacher_decode,
            seed=config.seed,
            backend="vllm_openai",
            model=None,
            tokenizer=None,
            base_url=base_url,
            api_key=config.teacher_api_key,
            teacher_mode=config.teacher_mode,
        )
    else:
        raise TeacherTranscriptError(f"unsupported teacher_backend: {config.teacher_backend!r}")

    if len(records) != budget:
        raise TeacherTranscriptError(f"transcript generation produced {len(records)} records, expected {budget}.")

    if existing_bundle is None:
        ledger = QueryLedger()
        ledger_start = 0
    else:
        ledger = QueryLedger.from_dict(existing_bundle.ledger.to_dict())
        ledger_start = len(existing_records)
    for order_index, record in enumerate(records[ledger_start:], start=ledger_start):
        ledger.record_success(
            QueryAttempt(
                query_id=record.query_id,
                prompt_id=record.prompt_id,
                order_index=order_index,
                attempt_index=1,
                status="success",
                created_at=record.created_at,
            )
        )
    manifest = save_transcript_bundle(
        bundle_dir=bundle_dir,
        prompt_pool_id=prompt_pool_id,
        ordering=ordering_obj,
        records=records,
        teacher_model_id=teacher_model_id,
        decode_config=teacher_decode,
        seed=config.seed,
        requested_budget=budget,
        ledger=ledger,
        prompt_pool_hash=prompt_pool_hash,
    )
    _write_teacher_build_sidecar(
        bundle_dir,
        config_hash=config_hash,
        config_path=config_path,
        query_pool_path=query_pool_path,
        query_ordering_path=query_ordering_path,
        manifest=manifest,
        ledger=ledger,
        config=config,
    )
    return {
        "status": "built",
        "bundle_dir": str(bundle_dir),
        "records": len(records),
        "budget": budget,
        "prompt_pool_id": prompt_pool_id,
        "prompt_pool_hash": prompt_pool_hash,
        "ordering_hash": ordering_hash,
        "config_hash": config_hash,
        "transcript_hash": manifest.transcript_hash,
        "manifest_path": str(bundle_dir / "manifest.json"),
        "ordering_path": str(bundle_dir / "ordering.json"),
        "ledger_path": str(bundle_dir / "ledger.json"),
        "sidecar_path": str(bundle_dir / "teacher_build.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--backend")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = build_teacher_transcript(
        config_path=Path(args.config),
        budget=args.budget,
        backend_override=args.backend,
        output_dir_override=Path(args.output_dir) if args.output_dir else None,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
