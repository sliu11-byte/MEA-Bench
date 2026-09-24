"""Persist and validate shared DuFFin teacher answers."""

from __future__ import annotations

import json
from pathlib import Path

from defenses.duffin.detector import build_probe_prompt, extract_choice, load_local_model, query_local_model_batch


def get_teacher_reference(path, probes, teacher_model, max_new_tokens, batch_size=4):
    path = Path(path)
    protocol = {
        "version": 1, "teacher_model": teacher_model,
        "max_new_tokens": max_new_tokens, "batch_size": batch_size, "do_sample": False,
        "prompt_format": "duffin_official_cot_v2", "probes": probes,
    }
    if path.exists():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact["protocol"] != protocol:
            raise ValueError("DuFFin teacher reference protocol/probes mismatch; use a new output directory")
        rows = artifact["rows"]
        if len(rows) != len(probes):
            raise ValueError("Incomplete DuFFin teacher reference")
        for probe, row in zip(probes, rows):
            _, allowed = build_probe_prompt(probe)
            if row["teacher_choice"] != extract_choice(row["teacher_response"], allowed_choices=allowed):
                raise ValueError("Corrupt DuFFin teacher reference choice")
        return rows, 0
    teacher, tokenizer = load_local_model(teacher_model)
    rows = []
    for start in range(0, len(probes), batch_size):
        batch = probes[start : start + batch_size]
        prepared = [build_probe_prompt(probe) for probe in batch]
        results = query_local_model_batch(
            teacher, tokenizer,
            [item[0] for item in prepared],
            allowed_choices=[item[1] for item in prepared],
            max_new_tokens=max_new_tokens,
        )
        rows.extend({"teacher_response": response, "teacher_choice": choice} for response, choice in results)
        print(f"[DuFFin teacher] {len(rows)}/{len(probes)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"protocol": protocol, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    del teacher, tokenizer
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, len(probes)
