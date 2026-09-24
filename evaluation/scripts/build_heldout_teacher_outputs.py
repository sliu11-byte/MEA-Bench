
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from typing import Any, Iterable



from attacks.core.manifest import stable_hash

PROMPT_FIELDS = ("prompt", "query", "prompt_text", "source_prompt")


class TeacherOutputError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TeacherOutputError(f"{path}:{line_number} must be a JSON object")
            yield row


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_id(row: dict[str, Any], index: int) -> str:
    for key in ("id", "query_id", "prompt_id"):
        if row.get(key) is not None:
            return str(row[key])
    return str(index)


def row_prompt(row: dict[str, Any], source: Path, index: int) -> str:
    for key in PROMPT_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise TeacherOutputError(f"{source}:{index} is missing one of {PROMPT_FIELDS}")


def load_prompts(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(iter_jsonl(path), start=1):
        rows.append({**row, "id": row_id(row, index), "prompt": row_prompt(row, path, index)})
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise TeacherOutputError(f"No prompt rows found in {path}")
    return rows


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row_id(row, index) for index, row in enumerate(iter_jsonl(path), start=1)}



def torch_dtype_from_name(name: str) -> Any:
    import torch

    normalized = str(name).lower().strip()
    if normalized == "auto":
        return "auto"
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise TeacherOutputError(f"Unsupported --torch-dtype: {name}")
    return aliases[normalized]


def load_local_hf_teacher(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch_dtype_from_name(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def render_local_prompt(tokenizer: Any, prompt: str, mode: str) -> dict[str, Any]:
    if mode == "chat" and getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        rendered = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    if isinstance(rendered, Mapping):
        inputs = dict(rendered)
    else:
        inputs = {"input_ids": rendered}
    if "attention_mask" not in inputs or inputs["attention_mask"] is None:
        import torch

        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return inputs


def generate_one_local_hf(
    prompt: str,
    args: argparse.Namespace,
    per_prompt_seed: int,
    model: Any,
    tokenizer: Any,
) -> tuple[str, dict[str, Any]]:
    import torch

    torch.manual_seed(per_prompt_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(per_prompt_seed)
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in render_local_prompt(tokenizer, prompt, args.mode).items()
    }
    input_ids = inputs["input_ids"]
    kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": args.max_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
    with torch.inference_mode():
        output_ids = model.generate(**kwargs)
    new_tokens = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, {
        "input_tokens": int(input_ids.shape[-1]),
        "output_tokens": int(new_tokens.shape[-1]),
        "total_tokens": int(input_ids.shape[-1] + new_tokens.shape[-1]),
    }

def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise TeacherOutputError(f"Teacher endpoint returned non-object JSON from {url}")
    return decoded


def extract_text(payload: dict[str, Any], mode: str) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TeacherOutputError("Teacher endpoint response is missing choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise TeacherOutputError("Teacher endpoint choice is not an object")
    if mode == "chat":
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TeacherOutputError("Chat response is missing choices[0].message.content")
        return message["content"].strip()
    text = choice.get("text")
    if not isinstance(text, str):
        raise TeacherOutputError("Completion response is missing choices[0].text")
    return text.strip()


def generate_one_openai(prompt: str, args: argparse.Namespace, per_prompt_seed: int) -> tuple[str, dict[str, Any]]:
    mode = args.mode.lower().strip()
    if mode == "chat":
        url = f"{args.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": args.request_model or args.teacher_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": per_prompt_seed,
        }
    elif mode == "completion":
        url = f"{args.base_url.rstrip('/')}/completions"
        payload = {
            "model": args.request_model or args.teacher_model,
            "prompt": prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": per_prompt_seed,
        }
    else:
        raise TeacherOutputError("--mode must be chat or completion")
    raw = post_json(url, payload, args.api_key, args.timeout)
    return extract_text(raw, mode), raw.get("usage") if isinstance(raw.get("usage"), dict) else {}


def build_teacher_outputs(args: argparse.Namespace) -> dict[str, Any]:
    prompts_path = Path(args.prompts_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else prompts_path.parent / "heldout_teacher_outputs.jsonl"
    manifest_path = Path(args.manifest_output).expanduser().resolve() if args.manifest_output else output_path.with_suffix(".manifest.json")
    prompts = load_prompts(prompts_path, args.limit)
    local_model = None
    local_tokenizer = None
    if args.backend == "local_hf":
        local_model, local_tokenizer = load_local_hf_teacher(args)
    done = existing_ids(output_path) if args.resume else set()
    pending = [row for row in prompts if str(row["id"]) not in done]

    written = 0
    failures = 0
    output_rows = []
    for order_index, row in enumerate(pending, start=len(done)):
        prompt_id = str(row["id"])
        prompt = str(row["prompt"])
        per_prompt_seed = args.seed + order_index
        last_error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                if args.backend == "local_hf":
                    assert local_model is not None and local_tokenizer is not None
                    teacher_response, usage = generate_one_local_hf(prompt, args, per_prompt_seed, local_model, local_tokenizer)
                else:
                    teacher_response, usage = generate_one_openai(prompt, args, per_prompt_seed)
                output_rows.append(
                    {
                        "id": prompt_id,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "teacher_response": teacher_response,
                        "teacher_model_id": args.teacher_model,
                        "request_model": args.request_model or args.teacher_model,
                        "mode": args.mode,
                        "seed": args.seed,
                        "order_index": order_index,
                        "created_at": utc_now_iso(),
                        "usage": usage,
                    }
                )
                written += 1
                break
            except (urllib.error.URLError, TimeoutError, TeacherOutputError) as exc:
                last_error = exc
                if attempt == args.max_attempts:
                    failures += 1
                    if args.fail_fast:
                        raise
                else:
                    time.sleep(min(args.max_backoff_seconds, args.initial_backoff_seconds * (2 ** (attempt - 1))))
        if last_error is not None and args.write_failures and (written + failures) % args.flush_every != 0:
            # Continue below; failures are represented only in the manifest by default.
            pass
        if len(output_rows) >= args.flush_every:
            append_jsonl(output_path, output_rows)
            output_rows.clear()
    if output_rows:
        append_jsonl(output_path, output_rows)

    completed_ids = existing_ids(output_path)
    manifest = {
        "schema_version": "heldout_teacher_outputs_v1",
        "prompts_jsonl": str(prompts_path),
        "output_jsonl": str(output_path),
        "teacher_model": args.teacher_model,
        "request_model": args.request_model or args.teacher_model,
        "backend": args.backend,
        "base_url": args.base_url,
        "torch_dtype": args.torch_dtype if args.backend == "local_hf" else None,
        "device_map": args.device_map if args.backend == "local_hf" else None,
        "mode": args.mode,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "prompt_count": len(prompts),
        "already_done": len(done),
        "written_this_run": written,
        "failures_this_run": failures,
        "completed_count": len(completed_ids),
        "completed_id_hash": stable_hash(sorted(completed_ids)),
        "created_at": utc_now_iso(),
    }
    write_json(manifest_path, manifest)
    return {"manifest_output": str(manifest_path), **manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate teacher outputs for held-out prompt JSONL rows.")
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--output-jsonl", help="Default: heldout_teacher_outputs.jsonl next to prompts.")
    parser.add_argument("--manifest-output")
    parser.add_argument("--teacher-model", required=True, help="Teacher model id/path for metadata and local_hf loading.")
    parser.add_argument("--backend", choices=("openai", "local_hf"), default="openai")
    parser.add_argument("--request-model", help="OpenAI-compatible model name to send; defaults to --teacher-model.")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--mode", choices=("chat", "completion"), default="chat")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--initial-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--limit", type=int, help="Optional smoke-test row limit.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--torch-dtype", default="auto", help="local_hf dtype: auto, bfloat16, float16, float32.")
    parser.add_argument("--device-map", default="auto", help="local_hf device_map passed to Transformers.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-failures", action="store_true", help="Reserved for future failure-row output; manifest always counts failures.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.backend == "openai" and not args.base_url:
        raise SystemExit("--base-url is required when --backend=openai")
    result = build_teacher_outputs(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
