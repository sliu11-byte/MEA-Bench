"""OpenAI-compatible response-rewriting proxy for online countermeasure runs."""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from countermeasures.waterpark_response import CountermeasureConfig, METHODS, build_rewriter
from defenses.core.io_utils import ensure_dir

LOGGER = logging.getLogger("countermeasure_proxy")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def post_json(url: str, payload: Mapping[str, Any], *, api_key: str, timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, *, api_key: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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
        return "\n".join(str(item) for item in prompt)
    return str(prompt)


class CountermeasureProxyState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.target_base_url = args.target_base_url.rstrip("/")
        self.api_key = args.api_key
        self.timeout = int(args.timeout)
        self.method = args.countermeasure
        self.config = CountermeasureConfig(
            method=args.countermeasure,
            input_path="online_openai_proxy",
            output_path=str(Path(args.output_dir) / "countermeasure_teacher_transcript.jsonl"),
            manifest_path=str(Path(args.output_dir) / "countermeasure_manifest.json"),
            lex=args.lex,
            order=args.order,
            sent_interval=args.sent_interval,
            no_ctx=not args.with_context,
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            device=args.device,
            seed=args.seed,
            source="WaterPark response-only online proxy",
        )
        self.output_dir = ensure_dir(args.output_dir)
        self.transcript_path = self.output_dir / "countermeasure_teacher_transcript.jsonl"
        self.manifest_path = self.output_dir / "countermeasure_manifest.json"
        self.log_lock = Lock()
        self.rewriter = build_rewriter(self.config)
        self.count = 0
        self.manifest_path.write_text(
            json.dumps(self.manifest_payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "countermeasure_manifest_v1",
            "created_at": utc_now_iso(),
            "countermeasure": f"waterpark_{self.method}",
            "countermeasure_family": "waterpark_response_only",
            "method": self.method,
            "mode": "online_openai_proxy",
            "source": self.config.source,
            "target_base_url": self.target_base_url,
            "output_transcript": str(self.transcript_path.resolve()),
            "num_output_records": self.count,
            "config": {
                "lex": self.config.lex,
                "order": self.config.order,
                "sent_interval": self.config.sent_interval,
                "no_ctx": self.config.no_ctx,
                "model_name": self.config.model_name,
                "tokenizer_name": self.config.tokenizer_name,
                "device": self.config.device,
                "seed": self.config.seed,
            },
            "artifacts": {"countermeasure_teacher_transcript": str(self.transcript_path.resolve())},
            "usage": {
                "teacher_queries": 0,
                "requires_clean_teacher_outputs": False,
                "requires_defense_key_or_detector": False,
            },
        }

    def rewrite(self, *, query: str, response: str, raw: Mapping[str, Any]) -> tuple[str, float]:
        started = time.time()
        if self.method == "dipper":
            rewritten = self.rewriter(
                response,
                query=query,
                lex=self.config.lex,
                order=self.config.order,
                sent_interval=self.config.sent_interval,
                no_ctx=self.config.no_ctx,
            )
        elif self.method == "translation":
            rewritten = self.rewriter(response)
        else:
            raise ValueError(f"unsupported countermeasure: {self.method}")
        elapsed = time.time() - started
        with self.log_lock:
            self.count += 1
            record = dict(raw)
            record.update(
                {
                    "query_id": str(raw.get("query_id") or f"online_{self.count:06d}"),
                    "query": query,
                    "source_response": response,
                    "response": rewritten,
                    "teacher_response": rewritten,
                    "countermeasure": f"waterpark_{self.method}",
                    "countermeasure_family": "waterpark_response_only",
                    "countermeasure_source": self.config.source,
                    "countermeasure_wall_time_seconds": elapsed,
                    "created_at": utc_now_iso(),
                }
            )
            with self.transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.manifest_path.write_text(
                json.dumps(self.manifest_payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return rewritten, elapsed


def write_json(handler: BaseHTTPRequestHandler, status: int, payload: Mapping[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_payload(message: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": "countermeasure_proxy_error"}}


class CountermeasureProxyHandler(BaseHTTPRequestHandler):
    server_version = "MEACountermeasureProxy/0.1"

    @property
    def state(self) -> CountermeasureProxyState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            try:
                payload = get_json(self.state.target_base_url + "/models", api_key=self.state.api_key, timeout=self.state.timeout)
                write_json(self, HTTPStatus.OK, payload)
            except Exception as exc:
                LOGGER.exception("failed to forward /models")
                write_json(self, HTTPStatus.BAD_GATEWAY, error_payload(str(exc)))
            return
        write_json(self, HTTPStatus.NOT_FOUND, error_payload(f"unknown path: {self.path}"))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except json.JSONDecodeError as exc:
            write_json(self, HTTPStatus.BAD_REQUEST, error_payload(f"invalid JSON: {exc}"))
            return

        if request.get("stream"):
            write_json(self, HTTPStatus.BAD_REQUEST, error_payload("streaming responses are not supported by the countermeasure proxy"))
            return

        route = self.path.rstrip("/")
        if route == "/v1/chat/completions":
            self._handle_chat(request)
            return
        if route == "/v1/completions":
            self._handle_completion(request)
            return
        write_json(self, HTTPStatus.NOT_FOUND, error_payload(f"unknown path: {self.path}"))

    def _forward(self, route: str, request: Mapping[str, Any]) -> dict[str, Any]:
        return post_json(self.state.target_base_url + route, request, api_key=self.state.api_key, timeout=self.state.timeout)

    def _handle_chat(self, request: Mapping[str, Any]) -> None:
        try:
            response_payload = self._forward("/chat/completions", request)
            query = prompt_from_chat(list(request.get("messages") or []))
            for choice_index, choice in enumerate(response_payload.get("choices") or []):
                message = choice.get("message") or {}
                content = message.get("content")
                if content is None:
                    continue
                rewritten, elapsed = self.state.rewrite(
                    query=query,
                    response=str(content),
                    raw={
                        "query_id": f"{response_payload.get('id') or uuid.uuid4().hex}:{choice_index}",
                        "route": "chat.completions",
                        "model": response_payload.get("model") or request.get("model"),
                    },
                )
                message["content"] = rewritten
                choice["message"] = message
                choice.setdefault("countermeasure", f"waterpark_{self.state.method}")
                choice.setdefault("countermeasure_wall_time_seconds", elapsed)
            response_payload["countermeasure"] = f"waterpark_{self.state.method}"
            write_json(self, HTTPStatus.OK, response_payload)
        except urllib.error.HTTPError as exc:
            write_json(self, exc.code, error_payload(exc.read().decode("utf-8", errors="replace")))
        except Exception as exc:
            LOGGER.exception("chat completion failed")
            write_json(self, HTTPStatus.BAD_GATEWAY, error_payload(str(exc)))

    def _handle_completion(self, request: Mapping[str, Any]) -> None:
        try:
            response_payload = self._forward("/completions", request)
            query = prompt_from_completion(request.get("prompt", ""))
            for choice_index, choice in enumerate(response_payload.get("choices") or []):
                text = choice.get("text")
                if text is None:
                    continue
                rewritten, elapsed = self.state.rewrite(
                    query=query,
                    response=str(text),
                    raw={
                        "query_id": f"{response_payload.get('id') or uuid.uuid4().hex}:{choice_index}",
                        "route": "completions",
                        "model": response_payload.get("model") or request.get("model"),
                    },
                )
                choice["text"] = rewritten
                choice.setdefault("countermeasure", f"waterpark_{self.state.method}")
                choice.setdefault("countermeasure_wall_time_seconds", elapsed)
            response_payload["countermeasure"] = f"waterpark_{self.state.method}"
            write_json(self, HTTPStatus.OK, response_payload)
        except urllib.error.HTTPError as exc:
            write_json(self, exc.code, error_payload(exc.read().decode("utf-8", errors="replace")))
        except Exception as exc:
            LOGGER.exception("completion failed")
            write_json(self, HTTPStatus.BAD_GATEWAY, error_payload(str(exc)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve an online WaterPark response-only countermeasure proxy.")
    parser.add_argument("--target-base-url", required=True, help="OpenAI-compatible defended teacher base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--countermeasure", choices=METHODS, required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--lex", type=int, default=40)
    parser.add_argument("--order", type=int, default=0)
    parser.add_argument("--sent-interval", type=int, default=3)
    parser.add_argument("--with-context", action="store_true")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    state = CountermeasureProxyState(args)
    server = ThreadingHTTPServer((args.host, args.port), CountermeasureProxyHandler)
    server.state = state  # type: ignore[attr-defined]
    LOGGER.info("countermeasure proxy ready at http://%s:%s/v1 -> %s", args.host, args.port, state.target_base_url)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
