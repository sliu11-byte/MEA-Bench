import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from defenses.oracle.generate_clean_teacher import generate


class CleanGenerationTests(unittest.TestCase):
    def test_resume_completed_responses_and_skip_finished_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries = root / "queries.jsonl"
            queries.write_text('\n'.join(json.dumps({"query_id": str(i), "query": f"query{i}"})
                                         for i in range(3)) + '\n')
            args = argparse.Namespace(output_dir=str(root / "out"), query_pool_path=str(queries),
                                      max_queries=3, defense_config="{}", teacher_model="teacher",
                                      device="cuda", temperature=0.0, top_p=1.0,
                                      max_new_tokens=1536, mode="chat")
            calls = []

            class Backend:
                def generate_one(self, query, *, query_id, request):
                    calls.append(query_id)
                    if query_id == "1" and calls == ["0", "1"]:
                        raise RuntimeError("interrupted")
                    return {"query_id": query_id, "query": query, "response": "answer"}

            factory = Mock(side_effect=lambda *args: Backend())
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                generate(args, factory)
            generate(args, factory)
            self.assertEqual(calls, ["0", "1", "1", "2"])
            self.assertEqual(factory.call_count, 2)
            generate(args, factory)
            self.assertEqual(factory.call_count, 2)
            transcript = root / "out" / "transcripts" / "defended_teacher.jsonl"
            rows = [json.loads(line) for line in transcript.read_text().splitlines()]
            self.assertEqual([row["query_id"] for row in rows], ["0", "1", "2"])
            args.max_new_tokens = 512
            with self.assertRaisesRegex(ValueError, "different configuration"):
                generate(args, factory)


if __name__ == "__main__":
    unittest.main()
