from __future__ import annotations

import argparse
import json
import sys
import socket
import tempfile
import threading
import unittest
import urllib.request
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from countermeasures.waterpark_response import (
    CountermeasureConfig,
    normalize_transcript_record,
    rewrite_transcript,
)
from countermeasures import waterpark_response


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps({"data": [{"id": "fake-teacher"}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path.rstrip("/") == "/v1/chat/completions":
            content = request["messages"][-1]["content"]
            payload = {
                "id": "fake-response",
                "model": request.get("model", "fake-teacher"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": f"answer::{content}"}}],
            }
        else:
            payload = {
                "id": "fake-response",
                "model": request.get("model", "fake-teacher"),
                "choices": [{"index": 0, "text": f"answer::{request.get('prompt', '')}"}],
            }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ServerThread:
    def __init__(self, server: ThreadingHTTPServer) -> None:
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)

    def __enter__(self) -> ThreadingHTTPServer:
        self.thread.start()
        return self.server

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class WaterParkResponseCountermeasureTest(unittest.TestCase):
    def test_normalizes_defense_transcript_fields(self) -> None:
        row = {
            "id": "r1",
            "prompt": "Explain distillation.",
            "teacher_response": "A compact answer.",
        }
        self.assertEqual(
            normalize_transcript_record(row, 0),
            {
                "query_id": "r1",
                "query": "Explain distillation.",
                "response": "A compact answer.",
            },
        )

    def test_rewrite_transcript_outputs_training_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "defended_teacher.jsonl"
            output = root / "cleaned_teacher.jsonl"
            manifest = root / "countermeasure_manifest.json"
            write_jsonl(
                source,
                [
                    {
                        "query_id": "q1",
                        "query": "What is model extraction?",
                        "response": "It is a watermark-bearing answer.",
                        "defense": "adfp",
                    }
                ],
            )

            def fake_rewriter(response: str, **kwargs) -> str:
                return f"rewritten::{response}::{kwargs.get('lex')}::{kwargs.get('order')}"

            report = rewrite_transcript(
                config=CountermeasureConfig(
                    method="dipper",
                    input_path=str(source),
                    output_path=str(output),
                    manifest_path=str(manifest),
                    lex=40,
                    order=0,
                ),
                rewriter=fake_rewriter,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["source_response"], "It is a watermark-bearing answer.")
            self.assertEqual(rows[0]["response"], "rewritten::It is a watermark-bearing answer.::40::0")
            self.assertEqual(rows[0]["teacher_response"], rows[0]["response"])
            self.assertEqual(rows[0]["countermeasure"], "waterpark_dipper")
            self.assertFalse(report["usage"]["requires_clean_teacher_outputs"])
            self.assertTrue(manifest.exists())

    def test_translation_fake_rewriter_gets_only_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "defended_teacher.jsonl"
            output = root / "cleaned_teacher.jsonl"
            manifest = root / "countermeasure_manifest.json"
            write_jsonl(source, [{"query_id": "q1", "query": "Q", "response": "hello"}])

            report = rewrite_transcript(
                config=CountermeasureConfig(
                    method="translation",
                    input_path=str(source),
                    output_path=str(output),
                    manifest_path=str(manifest),
                ),
                rewriter=lambda response, **_: response.upper(),
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["response"], "HELLO")
            self.assertEqual(report["countermeasure"], "waterpark_translation")

    def test_rewrite_resumes_after_completed_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output, manifest = root / "source.jsonl", root / "output.jsonl", root / "manifest.json"
            write_jsonl(source, [{"query_id": str(i), "query": f"q{i}", "response": f"r{i}"} for i in range(3)])
            config = CountermeasureConfig(method="translation", input_path=str(source),
                                          output_path=str(output), manifest_path=str(manifest))
            calls = []

            def interrupted(response, **kwargs):
                calls.append(response)
                if response == "r1" and calls == ["r0", "r1"]:
                    raise RuntimeError("interrupted")
                return "rewritten " + response

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                rewrite_transcript(config=config, rewriter=interrupted)
            rewrite_transcript(config=config, rewriter=interrupted)
            self.assertEqual(calls, ["r0", "r1", "r1", "r2"])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["query_id"] for row in rows], ["0", "1", "2"])

    def test_sentence_splitter_falls_back_without_nltk(self) -> None:
        if hasattr(waterpark_response._sent_tokenize, "_fallback_warned"):
            del waterpark_response._sent_tokenize._fallback_warned
        with unittest.mock.patch.dict(sys.modules, {"nltk": None, "nltk.tokenize": None}), \
             warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(waterpark_response._sent_tokenize("One. Two? Three!"),
                             ["One.", "Two?", "Three!"])
            waterpark_response._sent_tokenize("Four. Five.")
            self.assertEqual(len(caught), 1)


class OnlineCountermeasureProxyTest(unittest.TestCase):
    def test_chat_completion_is_rewritten_and_logged(self) -> None:
        import countermeasures.openai_proxy as proxy

        old_build_rewriter = proxy.build_rewriter
        proxy.build_rewriter = lambda config: (lambda response, **kwargs: f"rewritten::{response}")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                target_port = free_port()
                proxy_port = free_port()
                target = ThreadingHTTPServer(("127.0.0.1", target_port), FakeOpenAIHandler)
                with ServerThread(target):
                    args = argparse.Namespace(
                        target_base_url=f"http://127.0.0.1:{target_port}/v1",
                        api_key="EMPTY",
                        timeout=5,
                        countermeasure="dipper",
                        output_dir=tmp,
                        lex=40,
                        order=0,
                        sent_interval=3,
                        with_context=False,
                        model_name=None,
                        tokenizer_name=None,
                        device=None,
                        seed=None,
                    )
                    state = proxy.CountermeasureProxyState(args)
                    server = ThreadingHTTPServer(("127.0.0.1", proxy_port), proxy.CountermeasureProxyHandler)
                    server.state = state  # type: ignore[attr-defined]
                    with ServerThread(server):
                        payload = {
                            "model": "fake-teacher",
                            "messages": [{"role": "user", "content": "hello"}],
                            "stream": False,
                        }
                        req = urllib.request.Request(
                            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                            data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req, timeout=5) as response:
                            body = json.loads(response.read().decode("utf-8"))

                    self.assertEqual(body["choices"][0]["message"]["content"], "rewritten::answer::hello")
                    rows = [json.loads(line) for line in (Path(tmp) / "countermeasure_teacher_transcript.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(rows[0]["query"], "hello")
                    self.assertEqual(rows[0]["source_response"], "answer::hello")
                    self.assertEqual(rows[0]["response"], "rewritten::answer::hello")
        finally:
            proxy.build_rewriter = old_build_rewriter


if __name__ == "__main__":
    unittest.main()
