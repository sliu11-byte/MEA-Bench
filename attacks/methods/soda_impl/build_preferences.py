from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import read_jsonl, record_id, record_prompt, record_response, write_jsonl


def build_preference_records(
    teacher_records: list[dict[str, Any]],
    student_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    student_by_id = {record_id(record, idx): record for idx, record in enumerate(student_records)}
    output: list[dict[str, Any]] = []
    missing: list[str] = []
    for idx, teacher in enumerate(teacher_records):
        pid = record_id(teacher, idx)
        student = student_by_id.get(pid)
        if student is None:
            missing.append(pid)
            continue
        prompt = record_prompt(teacher)
        output.append(
            {
                "prompt_id": pid,
                "prompt": prompt,
                "chosen": record_response(teacher, ("teacher_response",)),
                "rejected": record_response(student, ("student_response",)),
                "method": "soda",
                "chosen_source": "teacher",
                "rejected_source": "student_zero_shot_static",
            }
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing {len(missing)} student negative records. First missing ids: {preview}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SODA DPO preference data.")
    parser.add_argument("--teacher-jsonl", required=True, help="JSONL with prompt and teacher response.")
    parser.add_argument("--student-jsonl", required=True, help="JSONL with prompt_id and student zero-shot response.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with prompt/chosen/rejected.")
    args = parser.parse_args()

    preferences = build_preference_records(read_jsonl(args.teacher_jsonl), read_jsonl(args.student_jsonl))
    write_jsonl(Path(args.output_jsonl), preferences)
    print(f"Wrote {len(preferences)} SODA preference records to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


