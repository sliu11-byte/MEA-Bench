"""Serve defense generators as an OpenAI-compatible teacher endpoint.

The attack pipeline should treat this service exactly like a normal teacher:
point ``--teacher-endpoint-url`` at ``http://host:port/v1`` and keep the attack
itself unchanged. The defense remains a black-box teacher from the attack side.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from defenses.core.generation import _post_openai_json, generate_openai_chat_responses
from defenses.core.io_utils import ensure_dir

LOGGER = logging.getLogger("defended_teacher")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json_arg(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    path = Path(raw).expanduser()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return json.loads(raw)


def openai_error(message: str, status: int = 400) -> tuple[int, Dict[str, Any]]:
    return status, {"error": {"message": message, "type": "defended_teacher_error"}}


def prompt_from_chat(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                        parts.append(str(item.get("text", "")))
                    elif isinstance(item, str):
                        parts.append(item)
                return "\n".join(p for p in parts if p)
            return str(content)
    return "\n".join(str(m.get("content", "")) for m in messages if m.get("content") is not None)


def prompt_from_completion(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        if not prompt:
            return ""
        return str(prompt[0])
    return str(prompt)


class BaseOracleBackend:
    name = "base"

    def generate_one(self, query: str, *, query_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def save_artifacts(self, artifacts_dir: Path) -> Dict[str, str]:
        return {}


def request_generation_config(request: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(defaults)
    if request.get("temperature") is not None:
        cfg["temperature"] = request["temperature"]
        cfg["tau"] = request["temperature"]
    if request.get("top_p") is not None:
        cfg["top_p"] = request["top_p"]
    max_tokens = request.get("max_tokens") or request.get("max_completion_tokens")
    if max_tokens is not None:
        cfg["max_new_tokens"] = max_tokens
        cfg["teacher_max_new_tokens"] = max_tokens
    return cfg


class CleanOpenAIBackend(BaseOracleBackend):
    name = "clean"

    def __init__(self, *, teacher_base_url: str, teacher_model: str, api_key: str = "EMPTY", config: Dict[str, Any]):
        self.teacher_base_url = teacher_base_url
        self.teacher_model = teacher_model
        self.api_key = api_key
        self.config = config

    def generate_one(self, query: str, *, query_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        cfg = request_generation_config(request, self.config)
        if request.get("_oracle_route") == "completions":
            payload = {
                "model": self.teacher_model,
                "prompt": query,
                "max_tokens": int(cfg.get("max_new_tokens", 256)),
                "temperature": float(cfg.get("temperature", 0.7)),
                "top_p": float(cfg.get("top_p", 0.95)),
                "n": 1,
            }
            result = _post_openai_json(
                self.teacher_base_url.rstrip("/") + "/completions",
                payload,
                api_key=self.api_key,
                timeout=int(cfg.get("timeout", 600)),
            )
            response = str((result.get("choices") or [{}])[0].get("text", ""))
        else:
            response = generate_openai_chat_responses(
                [query],
                base_url=self.teacher_base_url,
                model=self.teacher_model,
                api_key=self.api_key,
                system_prompt=cfg.get("system_prompt"),
                max_new_tokens=int(cfg.get("max_new_tokens", 256)),
                temperature=float(cfg.get("temperature", 0.7)),
                top_p=float(cfg.get("top_p", 0.95)),
                timeout=int(cfg.get("timeout", 600)),
                max_retries=int(cfg.get("max_retries", 3)),
            )[0]
        return {"query_id": query_id, "query": query, "response": response, "defense": "clean", "teacher_model": self.teacher_model}


class CleanLocalBackend(BaseOracleBackend):
    name = "clean"

    def __init__(self, teacher_model: str, device: Optional[str], config: Dict[str, Any]):
        from defenses.core.generation import load_causal_lm

        self.teacher_model = teacher_model
        self.model, self.tokenizer, _ = load_causal_lm(teacher_model, device=device)
        import torch

        device_map = getattr(self.model, "hf_device_map", {})
        LOGGER.info("clean teacher device map: %s", device_map)
        LOGGER.info("clean teacher dtype=%s generation_eos=%s tokenizer_eos=%s",
                    self.model.dtype, self.model.generation_config.eos_token_id,
                    self.tokenizer.eos_token_id)
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            LOGGER.info("GPU %s: %s free_GiB=%.2f total_GiB=%.2f allocated_GiB=%.2f",
                        index, torch.cuda.get_device_name(index), free / 1024**3,
                        total / 1024**3, torch.cuda.memory_allocated(index) / 1024**3)
        if any(str(value) in {"cpu", "disk"} for value in device_map.values()):
            LOGGER.warning("clean teacher has CPU/disk offload")
        self.config = config

    def generate_one(self, query: str, *, query_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        from defenses.core.generation import format_prompt, generate_responses

        cfg = request_generation_config(request, self.config)
        prompt = query if request.get("_oracle_route") == "completions" else format_prompt(
            self.tokenizer, query, system_prompt=cfg.get("system_prompt")
        )
        response = generate_responses(
            self.model, self.tokenizer, [prompt],
            max_new_tokens=int(cfg.get("max_new_tokens", 1536)),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
        )[0]
        return {"query_id": query_id, "query": query, "response": response,
                "defense": "clean", "teacher_model": self.teacher_model}


class GeneratorBackend(BaseOracleBackend):
    def __init__(self, generator: Any, *, defense: str, config: Dict[str, Any], artifacts_dir: Optional[Path] = None):
        self.generator = generator
        self.name = defense
        self.defense = defense
        self.config = config
        self._artifacts = {}
        if artifacts_dir is not None and hasattr(generator, "save_artifacts"):
            try:
                if defense == "radioactivity":
                    self._artifacts = generator.save_artifacts(artifacts_dir, build_filter=False)
                else:
                    self._artifacts = generator.save_artifacts(artifacts_dir)
            except TypeError:
                self._artifacts = generator.save_artifacts(artifacts_dir)

    def generate_one(self, query: str, *, query_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        cfg = request_generation_config(request, self.config)
        return self.generator.generate([{"query_id": query_id, "query": query}], config=cfg)[0]

    def save_artifacts(self, artifacts_dir: Path) -> Dict[str, str]:
        return dict(self._artifacts)


class ADFPOracleBackend(BaseOracleBackend):
    name = "adfp"

    def __init__(self, *, teacher_model: str, proxy_model: str, config: Dict[str, Any], artifacts_dir: Path):
        import torch
        from transformers import LogitsProcessorList
        from defenses.adfp.core import ADFPHash, ADFPLogitsProcessor, artifact_id, load_models

        self.torch = torch
        self.teacher_model = teacher_model
        self.proxy_model = proxy_model
        self.teacher, self.proxy, self.tokenizer = load_models(teacher_model, proxy_model)
        self.config = {
            "secret_key": config.get("secret_key", "adfp-benchmark-key-v1"),
            "gamma": float(config.get("gamma", 0.5)),
            "window_size": int(config.get("window_size", 2)),
            "strength_lambda": float(config.get("strength_lambda", config.get("strength", 140.0))),
            "temperature": float(config.get("temperature", 1.0)),
            "top_p": float(config.get("top_p", 1.0)),
            "max_new_tokens": int(config.get("max_new_tokens", 256)),
        }
        vocab_size = int(
            getattr(
                getattr(self.teacher, "config", None),
                "vocab_size",
                len(self.tokenizer.get_vocab()),
            )
        )
        self.fingerprint_hash = ADFPHash(
            vocab_size=vocab_size,
            secret_key=str(self.config["secret_key"]),
            gamma=float(self.config["gamma"]),
            window_size=int(self.config["window_size"]),
        )
        self.logits_processor = LogitsProcessorList([
            ADFPLogitsProcessor(
                self.proxy,
                self.fingerprint_hash,
                float(self.config["strength_lambda"]),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        ])
        fp_config = {
            "defense": "adfp",
            "type": "output_fingerprint",
            "teacher_model": teacher_model,
            "proxy_model": proxy_model,
            **self.config,
            "hash_scheme": "sha256_context_seed_v1",
        }
        self.fingerprint_artifact_id = artifact_id({**fp_config, "secret_key": self.config["secret_key"]})
        ensure_dir(artifacts_dir)
        hash_path = artifacts_dir / "hash_config.json"
        fp_path = artifacts_dir / "fingerprint_config.json"
        hash_config = self.fingerprint_hash.to_dict()
        hash_config["fingerprint_artifact_id"] = self.fingerprint_artifact_id
        hash_path.write_text(json.dumps(hash_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        fp_path.write_text(json.dumps({**fp_config, "fingerprint_artifact_id": self.fingerprint_artifact_id, "hash_config": str(hash_path)}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._artifacts = {"hash_config": str(hash_path.resolve()), "fingerprint_config": str(fp_path.resolve())}

    def generate_one(self, query: str, *, query_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        from defenses.adfp.core import build_prompt

        cfg = request_generation_config(request, self.config)
        full_prompt = build_prompt(self.tokenizer, query)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.teacher.device)
        temperature = float(cfg.get("temperature", self.config["temperature"]))
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": int(cfg.get("max_new_tokens", self.config["max_new_tokens"])),
            "do_sample": do_sample,
            "logits_processor": self.logits_processor,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update({"temperature": temperature, "top_p": float(cfg.get("top_p", self.config["top_p"]))})
        with self.torch.no_grad():
            generated_ids = self.teacher.generate(**inputs, **generation_kwargs)
        new_tokens = generated_ids[0, inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {
            "query_id": query_id,
            "query": query,
            "response": response,
            "teacher_response": response,
            "defense": "adfp",
            "defense_role": "output_fingerprint_generator",
            "teacher_model": self.teacher_model,
            "proxy_model": self.proxy_model,
            "adfp_config": dict(self.config),
            "fingerprint_artifact_id": self.fingerprint_artifact_id,
        }

    def save_artifacts(self, artifacts_dir: Path) -> Dict[str, str]:
        return dict(self._artifacts)


class DOGeOracleBackend(BaseOracleBackend):
    name = "doge"

    def __init__(self, *, checkpoint: str, config: Dict[str, Any]):
        import torch
        from defenses.doge.run import load_defended_teacher

        self.torch = torch
        self.checkpoint = checkpoint
        self.model, self.tokenizer = load_defended_teacher(checkpoint)
        cfg_path = Path(checkpoint) / "doge_config.json"
        self.doge_config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        self.config = {"max_new_tokens": 256, "temperature": 0.0, **config}

    def _build_prompt(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"### User:\n{query}\n\n### Assistant:\n"

    def generate_one(self, query: str, *, query_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        cfg = request_generation_config(request, self.config)
        full_prompt = self._build_prompt(query)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        temperature = float(cfg.get("temperature", 0.0))
        do_sample = temperature > 0
        kwargs = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = float(cfg.get("top_p", 1.0))
        with self.torch.no_grad():
            generated_ids = self.model.generate(**inputs, **kwargs)
        response = self.tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        return {
            "query_id": query_id,
            "query": query,
            "response": response,
            "teacher_response": response,
            "defense": "doge",
            "teacher_model": self.doge_config.get("teacher_model", self.checkpoint),
            "doge_teacher_checkpoint": self.checkpoint,
            "doge_config": self.doge_config,
        }


def build_backend(args: argparse.Namespace, artifacts_dir: Path) -> BaseOracleBackend:
    cfg = load_json_arg(args.defense_config)
    defense = args.defense
    if defense == "clean":
        if not args.teacher_base_url:
            return CleanLocalBackend(args.teacher_model, args.device, cfg)
        return CleanOpenAIBackend(
            teacher_base_url=args.teacher_base_url,
            teacher_model=args.teacher_request_model or args.teacher_model,
            api_key=args.teacher_api_key,
            config=cfg,
        )
    if defense == "ginsew":
        from defenses.ginsew.generator import GinsewGenerator

        return GeneratorBackend(
            GinsewGenerator(args.teacher_model, device=args.device, system_prompt=cfg.get("system_prompt"), watermark_config=cfg),
            defense="ginsew",
            config=cfg,
            artifacts_dir=artifacts_dir,
        )
    if defense == "radioactivity":
        from defenses.radioactivity.generator import RadioactivityGenerator

        return GeneratorBackend(
            RadioactivityGenerator(args.teacher_model, device=args.device, system_prompt=cfg.get("system_prompt"), watermark_config=cfg),
            defense="radioactivity",
            config=cfg,
            artifacts_dir=artifacts_dir,
        )
    if defense == "ads":
        from defenses.ads.generator import ADSGenerator

        if not args.teacher_model:
            raise SystemExit("ADS oracle requires --teacher-model")
        if float(cfg.get("lam", 0.0)) != 0 and not (args.grad_path or cfg.get("grad_path")):
            raise SystemExit("ADS oracle with lam>0 requires --grad-path or defense_config.grad_path precomputed before serving")
        return GeneratorBackend(
            ADSGenerator(
                args.teacher_model,
                proxy_student=args.proxy_model or cfg.get("proxy_student"),
                grad_path=args.grad_path or cfg.get("grad_path"),
                device=args.device,
                system_prompt=cfg.get("system_prompt"),
            ),
            defense="ads",
            config=cfg,
            artifacts_dir=artifacts_dir,
        )
    if defense == "trace_rewriting":
        from defenses.trace_rewriting.generator import TraceRewritingGenerator

        if not args.teacher_model:
            raise SystemExit("Trace Rewriting oracle requires --teacher-model")
        return GeneratorBackend(
            TraceRewritingGenerator(
                teacher_model=args.teacher_model,
                rewriter_model=args.rewriter_model or args.teacher_model,
                device=args.device,
                teacher_system_prompt=cfg.get("teacher_system_prompt"),
                rewriter_system_prompt=cfg.get("rewriter_system_prompt"),
                load_teacher=args.teacher_base_url is None,
                teacher_backend="openai_compatible" if args.teacher_base_url else "local_hf",
                teacher_base_url=args.teacher_base_url,
                teacher_api_key=args.teacher_api_key,
                teacher_request_model=args.teacher_request_model or args.teacher_model,
                rewriter_backend=args.rewriter_backend,
                rewriter_base_url=args.rewriter_base_url,
                rewriter_api_key=args.rewriter_api_key,
                rewriter_request_model=args.rewriter_request_model or args.rewriter_model or args.teacher_model,
            ),
            defense="trace_rewriting",
            config=cfg,
            artifacts_dir=artifacts_dir,
        )
    if defense == "adfp":
        if not args.teacher_model or not args.proxy_model:
            raise SystemExit("ADFP oracle requires --teacher-model and --proxy-model")
        return ADFPOracleBackend(teacher_model=args.teacher_model, proxy_model=args.proxy_model, config=cfg, artifacts_dir=artifacts_dir)
    if defense == "doge":
        if not args.doge_checkpoint:
            raise SystemExit("DOGe oracle requires --doge-checkpoint")
        return DOGeOracleBackend(checkpoint=args.doge_checkpoint, config=cfg)
    raise SystemExit(f"unknown defense {defense!r}")


class OracleState:
    def __init__(self, *, backend: BaseOracleBackend, output_dir: Path, model_name: str, defense: str, run_id: str):
        self.backend = backend
        self.output_dir = output_dir
        self.model_name = model_name
        self.defense = defense
        self.run_id = run_id
        self.lock = Lock()
        self.count = 0
        self.query_log = output_dir / "teacher_received_queries.jsonl"
        self.transcript_log = output_dir / "defended_teacher_transcript.jsonl"

    def next_query_id(self) -> str:
        with self.lock:
            self.count += 1
            return f"{self.run_id}_{self.count:08d}"

    def append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        with self.lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


class OpenAIHandler(BaseHTTPRequestHandler):
    server_version = "DefendedTeacher/0.1"

    @property
    def state(self) -> OracleState:
        return self.server.state  # type: ignore[attr-defined]

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"/v1/models", "/models"}:
            self._send_json(200, {"object": "list", "data": [{"id": self.state.model_name, "object": "model", "owned_by": "defended-teacher"}]})
            return
        self._send_json(*openai_error(f"unknown route: {self.path}", HTTPStatus.NOT_FOUND))

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8")
            request = json.loads(body or "{}")
            route = self.path.rstrip("/")
            if route.endswith("/chat/completions"):
                status, payload = self.handle_chat_completion(request)
            elif route.endswith("/completions"):
                status, payload = self.handle_completion(request)
            else:
                status, payload = openai_error(f"unknown route: {self.path}", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            LOGGER.exception("request failed")
            status, payload = openai_error(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)
        self._send_json(int(status), payload)

    def handle_chat_completion(self, request: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        messages = request.get("messages") or []
        if not isinstance(messages, list):
            return openai_error("messages must be a list")
        query = prompt_from_chat(messages)
        record = self._generate_record(query, request, route="chat.completions")
        now = int(time.time())
        return 200, {
            "id": f"chatcmpl-{record['query_id']}",
            "object": "chat.completion",
            "created": now,
            "model": request.get("model") or self.state.model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": record.get("response", "")}, "finish_reason": "stop"}],
        }

    def handle_completion(self, request: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        query = prompt_from_completion(request.get("prompt", ""))
        record = self._generate_record(query, request, route="completions")
        now = int(time.time())
        return 200, {
            "id": f"cmpl-{record['query_id']}",
            "object": "text_completion",
            "created": now,
            "model": request.get("model") or self.state.model_name,
            "choices": [{"index": 0, "text": record.get("response", ""), "finish_reason": "stop"}],
        }

    def _generate_record(self, query: str, request: Dict[str, Any], *, route: str) -> Dict[str, Any]:
        query_id = self.state.next_query_id()
        request_model = request.get("model") or self.state.model_name
        request = {**request, "_oracle_route": route}
        query_record = {
            "query_id": query_id,
            "query": query,
            "request_model": request_model,
            "route": route,
            "defense": self.state.defense,
            "received_at": utc_now_iso(),
        }
        self.state.append_jsonl(self.state.query_log, query_record)
        received = time.monotonic()
        with self.state.lock:
            LOGGER.info("request %s generation starting: queue_seconds=%.3f", query_id,
                        time.monotonic() - received)
            record = self.server.generation_worker.submit(
                self.state.backend.generate_one, query, query_id=query_id, request=request
            ).result()
        LOGGER.info("request %s completed: total_seconds=%.3f", query_id, time.monotonic() - received)
        record = {**record, "query_id": query_id, "query": query, "request_model": request_model, "route": route, "served_at": utc_now_iso()}
        self.state.append_jsonl(self.state.transcript_log, record)
        return record

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


class OracleHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], state: OracleState):
        super().__init__(server_address, handler_class)
        self.state = state
        # Reuse one inference thread instead of initializing GPU runtime per request thread.
        self.generation_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oracle-generation")

    def server_close(self) -> None:
        super().server_close()
        self.generation_worker.shutdown(wait=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve a defense generator as an OpenAI-compatible teacher endpoint")
    p.add_argument("--defense", required=True, choices=["clean", "ginsew", "radioactivity", "ads", "trace_rewriting", "adfp", "doge"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-id", default=None)
    p.add_argument("--served-model-name", default=None, help="Model id exposed to attack clients; defaults to defended-<defense>")
    p.add_argument("--defense-config", default=None, help="Inline JSON or path to JSON config")
    p.add_argument("--device", default=None)
    p.add_argument("--teacher-model", default=None)
    p.add_argument("--teacher-base-url", default=None, help="Upstream OpenAI-compatible clean teacher URL for passthrough/trace_rewriting")
    p.add_argument("--teacher-request-model", default=None)
    p.add_argument("--teacher-api-key", default="EMPTY")
    p.add_argument("--proxy-model", default=None, help="Proxy/student model used by ADS or ADFP")
    p.add_argument("--grad-path", default=None, help="Precomputed ADS gradient path")
    p.add_argument("--rewriter-model", default=None)
    p.add_argument("--rewriter-backend", choices=["local_hf", "openai_compatible"], default="local_hf")
    p.add_argument("--rewriter-base-url", default=None)
    p.add_argument("--rewriter-request-model", default=None)
    p.add_argument("--rewriter-api-key", default="EMPTY")
    p.add_argument("--doge-checkpoint", default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(asctime)s] %(levelname)s %(message)s")
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    artifacts_dir = ensure_dir(output_dir / "artifacts")
    model_name = args.served_model_name or f"defended-{args.defense}"
    backend = build_backend(args, artifacts_dir)
    manifest = {
        "schema_version": "defended_teacher_oracle_v1",
        "run_id": run_id,
        "defense": args.defense,
        "served_model_name": model_name,
        "created_at": utc_now_iso(),
        "base_url": f"http://{args.host}:{args.port}/v1",
        "logs": {
            "teacher_received_queries": str((output_dir / "teacher_received_queries.jsonl").resolve()),
            "defended_teacher_transcript": str((output_dir / "defended_teacher_transcript.jsonl").resolve()),
        },
        "artifacts": backend.save_artifacts(artifacts_dir),
        "config": load_json_arg(args.defense_config),
    }
    (output_dir / "oracle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    state = OracleState(backend=backend, output_dir=output_dir, model_name=model_name, defense=args.defense, run_id=run_id)
    server = OracleHTTPServer((args.host, args.port), OpenAIHandler, state)
    LOGGER.info("serving defended teacher: defense=%s model=%s url=http://%s:%s/v1", args.defense, model_name, args.host, args.port)
    LOGGER.info("logs: %s", output_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
