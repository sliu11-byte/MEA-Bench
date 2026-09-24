import tempfile
import unittest
from pathlib import Path

from attacks.methods.stage1_budget_impl.stage1_transcript import (
    TranscriptRecord,
    read_transcript_jsonl,
    write_transcript_jsonl,
)


class Stage1TranscriptJsonlTests(unittest.TestCase):
    def test_unicode_line_separators_remain_inside_json_strings(self):
        records = [
            TranscriptRecord(
                prompt_id="p0",
                prompt_text="prompt\u0085continued",
                teacher_response="first\u2028second\u2029third",
                teacher_model_id="teacher",
                decode_config={"strategy": "greedy", "settings": {}},
                seed=1,
                order_index=0,
                query_id="q0",
                query_status="success",
                created_at="2026-09-21T00:00:00+00:00",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcript.jsonl"
            write_transcript_jsonl(path, records)
            loaded = read_transcript_jsonl(path)

        self.assertEqual(loaded, records)


if __name__ == "__main__":
    unittest.main()
