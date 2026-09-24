"""Response-only WaterPark countermeasures for MEA transcripts.

This module vendors the two WaterPark attacks that match the MEA black-box
threat model: DIPPER paraphrasing and back-translation. Both consume only the
defended teacher response text and write a cleaned response that can be passed
to the existing attack pipeline via ``--teacher-transcript-path``.

The model calls are adapted from WaterPark:
  - watermark_reliability_release/utils/dipper_attack_pipeline.py
  - watermark_reliability_release/utils/generate_translation.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from defenses.core.io_utils import read_jsonl, write_jsonl
except ModuleNotFoundError:  # pragma: no cover - convenience for direct reuse.
    def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                records.append(payload)
        return records

    def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        return count


QUERY_ID_KEYS = ("query_id", "id", "qid", "uid", "example_id", "idx")
QUERY_TEXT_KEYS = ("query", "prompt", "rendered_prompt", "logical_prompt", "instruction", "question", "problem", "text", "input")
RESPONSE_KEYS = ("response", "teacher_response", "output", "completion", "answer", "final_answer", "w_wm_output")
METHODS = ("dipper", "translation")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pick(payload: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _sent_tokenize(text: str) -> list[str]:
    try:
        try:
            from nltk.tokenize import sent_tokenize

            return sent_tokenize(text)
        except LookupError:
            import nltk

            nltk.download("punkt", quiet=True)
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
            return sent_tokenize(text)
    except (ImportError, LookupError):
        if not getattr(_sent_tokenize, "_fallback_warned", False):
            warnings.warn("NLTK/Punkt is unavailable; using the built-in sentence splitter for DIPPER.")
            _sent_tokenize._fallback_warned = True
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
        return sentences or ([text.strip()] if text.strip() else [])
        return sent_tokenize(text)


def normalize_transcript_record(raw: Mapping[str, Any], index: int) -> dict[str, str]:
    query_id = _pick(raw, QUERY_ID_KEYS)
    query = _pick(raw, QUERY_TEXT_KEYS)
    response = _pick(raw, RESPONSE_KEYS)
    if query_id is None:
        query_id = f"countermeasure_{index:06d}"
    if query is None:
        raise ValueError(f"record {index} missing query field; tried {QUERY_TEXT_KEYS}")
    if response is None:
        raise ValueError(f"record {index} missing response field; tried {RESPONSE_KEYS}")
    return {
        "query_id": str(query_id),
        "query": str(query),
        "response": str(response),
    }


class DipperRewriter:
    """DIPPER paraphraser adapted from WaterPark."""

    def __init__(
        self,
        *,
        model_name: str = "kalpeshk2011/dipper-paraphraser-xxl",
        tokenizer_name: str = "google/t5-v1_1-xxl",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.torch = torch
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def __call__(
        self,
        response: str,
        *,
        query: str = "",
        lex: int = 40,
        order: int = 0,
        sent_interval: int = 3,
        no_ctx: bool = True,
        do_sample: bool = True,
        top_p: float = 0.75,
        max_length: int = 512,
    ) -> str:
        if lex not in {0, 20, 40, 60, 80, 100}:
            raise ValueError("lex must be one of 0, 20, 40, 60, 80, 100")
        if order not in {0, 20, 40, 60, 80, 100}:
            raise ValueError("order must be one of 0, 20, 40, 60, 80, 100")

        input_text = " ".join(response.split())
        sentences = _sent_tokenize(input_text)
        prefix = " ".join(query.replace("\n", " ").split())
        output_text = ""
        lex_code = int(100 - lex)
        order_code = int(100 - order)

        for sent_idx in range(0, len(sentences), sent_interval):
            window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
            if no_ctx:
                prompt = f"lexical = {lex_code}, order = {order_code} <sent> {window} </sent>"
            else:
                prompt = f"lexical = {lex_code}, order = {order_code} {prefix} <sent> {window} </sent>"
            inputs = self.tokenizer([prompt], return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=None,
                    max_length=max_length,
                )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            prefix += " " + decoded
            output_text += " " + decoded
        return output_text.strip()


class TranslationRewriter:
    """English -> French -> English back-translation adapted from WaterPark."""

    def __init__(
        self,
        *,
        model_name: str = "facebook/seamless-m4t-v2-large",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SeamlessM4Tv2Model.from_pretrained(model_name)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def __call__(
        self,
        response: str,
        *,
        src_lang: str = "eng",
        pivot_lang: str = "fra",
        **_: Any,
    ) -> str:
        text = response.strip()
        if not text:
            return text
        encoded = self.processor(text=text, src_lang=src_lang, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            pivot_tokens = self.model.generate(
                **encoded,
                tgt_lang=pivot_lang,
                generate_speech=False,
            )
        pivot_text = self.processor.decode(pivot_tokens[0].tolist()[0], skip_special_tokens=True)

        encoded_pivot = self.processor(text=pivot_text, src_lang=pivot_lang, return_tensors="pt")
        encoded_pivot = {key: value.to(self.device) for key, value in encoded_pivot.items()}
        with self.torch.inference_mode():
            back_tokens = self.model.generate(
                **encoded_pivot,
                tgt_lang=src_lang,
                generate_speech=False,
            )
        return self.processor.decode(back_tokens[0].tolist()[0], skip_special_tokens=True).strip()


@dataclass(frozen=True)
class CountermeasureConfig:
    method: str
    input_path: str
    output_path: str
    manifest_path: str
    lex: int = 40
    order: int = 0
    sent_interval: int = 3
    no_ctx: bool = True
    model_name: str | None = None
    tokenizer_name: str | None = None
    device: str | None = None
    seed: int | None = None
    limit: int | None = None
    source: str = "WaterPark response-only attack"
    extra: dict[str, Any] = field(default_factory=dict)


def build_rewriter(config: CountermeasureConfig) -> Callable[..., str]:
    if config.method == "dipper":
        return DipperRewriter(
            model_name=config.model_name or "kalpeshk2011/dipper-paraphraser-xxl",
            tokenizer_name=config.tokenizer_name or "google/t5-v1_1-xxl",
            device=config.device,
        )
    if config.method == "translation":
        return TranslationRewriter(
            model_name=config.model_name or "facebook/seamless-m4t-v2-large",
            device=config.device,
        )
    raise ValueError(f"unknown WaterPark response-only method: {config.method}")


def rewrite_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: CountermeasureConfig,
    rewriter: Callable[..., str],
) -> list[dict[str, Any]]:
    selected = list(records if config.limit is None else records[: max(0, config.limit)])
    outputs: list[dict[str, Any]] = []
    for index, raw in enumerate(selected):
        normalized = normalize_transcript_record(raw, index)
        started = time.time()
        if config.method == "dipper":
            rewritten = rewriter(
                normalized["response"],
                query=normalized["query"],
                lex=config.lex,
                order=config.order,
                sent_interval=config.sent_interval,
                no_ctx=config.no_ctx,
            )
        elif config.method == "translation":
            rewritten = rewriter(normalized["response"])
        else:
            raise ValueError(f"unsupported method: {config.method}")

        out = dict(raw)
        out["query_id"] = normalized["query_id"]
        out["query"] = normalized["query"]
        out["source_response"] = normalized["response"]
        out["response"] = rewritten
        out["teacher_response"] = rewritten
        out["countermeasure"] = f"waterpark_{config.method}"
        out["countermeasure_family"] = "waterpark_response_only"
        out["countermeasure_source"] = config.source
        out["countermeasure_wall_time_seconds"] = time.time() - started
        outputs.append(out)
    return outputs


def rewrite_transcript(
    *,
    config: CountermeasureConfig,
    rewriter: Callable[..., str] | None = None,
) -> dict[str, Any]:
    input_path = Path(config.input_path).expanduser().resolve()
    output_path = Path(config.output_path).expanduser().resolve()
    manifest_path = Path(config.manifest_path).expanduser().resolve()
    if config.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {config.method!r}")
    records = read_jsonl(input_path)
    if not records:
        raise ValueError(f"input transcript is empty: {input_path}")

    config_path = manifest_path.parent / "countermeasure_config.json"
    config_payload = asdict(config)
    completed = read_jsonl(output_path) if output_path.exists() else []
    if completed and (not config_path.exists() or json.loads(config_path.read_text()) != config_payload):
        raise ValueError("existing countermeasure transcript was generated with a different configuration")
    if len(completed) > len(records):
        raise ValueError("existing countermeasure transcript is longer than its input transcript")
    for index, (source, output) in enumerate(zip(records, completed)):
        expected = normalize_transcript_record(source, index)
        if output.get("query_id") != expected["query_id"] or output.get("source_response") != expected["response"]:
            raise ValueError("existing countermeasure transcript does not match its input transcript")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[countermeasure] completed={len(completed)}/{len(records)} output={output_path}", flush=True)
    if len(completed) < len(records):
        active_rewriter = rewriter or build_rewriter(config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for index in range(len(completed), len(records)):
                started = time.monotonic()
                print(f"[countermeasure] rewriting {index + 1}/{len(records)}", flush=True)
                output = rewrite_records(records[index:index + 1], config=config, rewriter=active_rewriter)[0]
                expected = normalize_transcript_record(records[index], index)
                output["query_id"] = expected["query_id"]
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[countermeasure] completed {index + 1}/{len(records)} in {time.monotonic()-started:.1f}s", flush=True)
    outputs = read_jsonl(output_path)

    manifest = {
        "schema_version": "countermeasure_manifest_v1",
        "created_at": _utc_now(),
        "countermeasure": f"waterpark_{config.method}",
        "countermeasure_family": "waterpark_response_only",
        "method": config.method,
        "source": config.source,
        "input_transcript": str(input_path),
        "output_transcript": str(output_path),
        "num_input_records": len(records),
        "num_output_records": len(outputs),
        "config": asdict(config),
        "artifacts": {
            "cleaned_teacher_transcript": str(output_path),
        },
        "usage": {
            "teacher_queries": 0,
            "requires_clean_teacher_outputs": False,
            "requires_defense_key_or_detector": False,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _default_output(input_path: Path, method: str) -> Path:
    return Path("outputs") / "countermeasures" / f"waterpark_{method}" / input_path.stem / "transcripts" / "cleaned_teacher.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WaterPark response-only countermeasures on an MEA defended transcript."
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--input", required=True, help="MEA defended_teacher.jsonl or query/response JSONL.")
    parser.add_argument("--output", default=None, help="Output cleaned_teacher.jsonl path.")
    parser.add_argument("--manifest", default=None, help="Output countermeasure_manifest.json path.")
    parser.add_argument("--lex", type=int, default=40, help="DIPPER lexical diversity. MEA default: 40.")
    parser.add_argument("--order", type=int, default=0, help="DIPPER order diversity. MEA default: 0.")
    parser.add_argument("--sent-interval", type=int, default=3)
    parser.add_argument("--with-context", action="store_true", help="Pass query context into DIPPER.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser() if args.output else _default_output(input_path, args.method)
    manifest_path = Path(args.manifest).expanduser() if args.manifest else output_path.parent.parent / "countermeasure_manifest.json"
    config = CountermeasureConfig(
        method=args.method,
        input_path=str(input_path),
        output_path=str(output_path),
        manifest_path=str(manifest_path),
        lex=args.lex,
        order=args.order,
        sent_interval=args.sent_interval,
        no_ctx=not args.with_context,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        seed=args.seed,
        limit=args.limit,
    )
    manifest = rewrite_transcript(config=config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
