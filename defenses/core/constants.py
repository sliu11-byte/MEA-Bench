"""Canonical strings mandated by the ``*复现范围.md`` spec docs.

Each doc's "Generator 输出" / "Run Manifest" section pins down an exact
``defense_role`` (per-record) and manifest ``type`` (per-run) string for its
defense. Centralizing them here keeps every generator/run CLI in sync with
the docs instead of re-typing (and risking drift on) the literal strings.
"""

from __future__ import annotations

DEFENSE_ROLE = {
    "adfp": "output_fingerprint_generator",
    "ads": "anti_distillation_generator",
    "doge": "anti_distillation_generator",
    "ginsew": "output_watermark_generator",
    "radioactivity": "radioactive_watermark_generator",
    "trace_rewriting": "anti_distillation_generator",
}

DEFENSE_TYPE = {
    "adfp": "output_fingerprint",
    "ads": "anti_distillation_generator",
    "doge": "anti_distillation_generator",
    "ginsew": "output_watermark",
    "radioactivity": "radioactive_watermark",
    "trace_rewriting": "anti_distillation_generator",
}
