from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .common import read_jsonl, record_id, record_prompt, record_response, write_jsonl


def build_preference_records(
    teacher_records: list[dict[str, Any]],
    student_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    student_by_id = {record_id(record, idx): record for idx, record in enumerate(student_records)}
    output: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid_teacher: list[str] = []
    invalid_student: list[str] = []
    for idx, teacher in enumerate(teacher_records):
        pid = record_id(teacher, idx)
        student = student_by_id.get(pid)
        if student is None:
            missing.append(pid)
            continue
        prompt = record_prompt(teacher)
        try:
            chosen = record_response(teacher, ("teacher_response",))
        except ValueError:
            invalid_teacher.append(pid)
            continue
        try:
            rejected = record_response(student, ("student_response",))
        except ValueError:
            invalid_student.append(pid)
            continue
        output.append(
            {
                "prompt_id": pid,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "method": "soda",
                "chosen_source": "teacher",
                "rejected_source": "student_zero_shot_static",
            }
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing {len(missing)} student negative records. First missing ids: {preview}")
    stats = {
        "schema_version": "soda_preference_stats_v1",
        "teacher_records": len(teacher_records),
        "student_records": len(student_records),
        "written_preferences": len(output),
        "skipped_records": len(invalid_teacher) + len(invalid_student),
        "invalid_teacher_responses": len(invalid_teacher),
        "invalid_student_responses": len(invalid_student),
        "invalid_teacher_prompt_ids": invalid_teacher,
        "invalid_student_prompt_ids": invalid_student,
    }
    return output, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SODA DPO preference data.")
    parser.add_argument("--teacher-jsonl", required=True, help="JSONL with prompt and teacher response.")
    parser.add_argument("--student-jsonl", required=True, help="JSONL with prompt_id and student zero-shot response.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with prompt/chosen/rejected.")
    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    preferences, stats = build_preference_records(read_jsonl(args.teacher_jsonl), read_jsonl(args.student_jsonl))
    write_jsonl(output_path, preferences)
    stats_path = output_path.with_name("preference_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if stats["skipped_records"]:
        print(
            "Skipped "
            f"{stats['skipped_records']} unusable SODA pairs "
            f"(teacher={stats['invalid_teacher_responses']}, student={stats['invalid_student_responses']}).",
            file=sys.stderr,
        )
    print(f"Wrote {len(preferences)} SODA preference records to {args.output_jsonl}")
    print(f"Wrote SODA preference statistics to {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


