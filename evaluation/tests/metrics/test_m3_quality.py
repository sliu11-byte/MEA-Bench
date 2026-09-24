import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics.m3_quality import (
    aggregate_pairwise_judgments,
    joint_tokenization,
    rep_n_from_tokens,
    repetition_report,
    summarize_ppl_against_human_interval,
    tokenize,
)


JUDGE_PROTOCOL = {
    "judge_model": "judge-v1",
    "judge_revision": "2026-07-13",
    "rubric_id": "matrix-m3-v1",
}


def judgment(
    row_id,
    *,
    forward_winner,
    reverse_winner,
    candidate_text="candidate text",
    reference_text="reference text",
):
    return {
        "id": row_id,
        "candidate_text": candidate_text,
        "reference_text": reference_text,
        "forward": {"candidate_position": "A", "winner": forward_winner},
        "reverse": {"candidate_position": "B", "winner": reverse_winner},
    }


class M3MetricsTest(unittest.TestCase):
    def test_ppl_rejects_cross_boundary_oracle_token(self):
        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                self.text = text
                return {
                    "input_ids": [11, 12],
                    "offset_mapping": [(0, 3), (3, 5)],
                }

        tokenizer = FakeTokenizer()
        with self.assertRaisesRegex(ValueError, "splits an oracle token"):
            joint_tokenization(tokenizer, "ab", "c d", "x")
        self.assertEqual(tokenizer.text, "abc d")

    def test_ppl_scores_only_token_aligned_continuation(self):
        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                return {
                    "input_ids": [11, 12],
                    "offset_mapping": [(0, 2), (2, 4)],
                }

        token_ids, score_mask = joint_tokenization(
            FakeTokenizer(), "ab", "cd", "x"
        )
        self.assertEqual(token_ids, [11, 12])
        self.assertEqual(score_mask, [False, True])

    def test_ppl_rejects_empty_continuation(self):
        with self.assertRaisesRegex(ValueError, "empty continuation"):
            joint_tokenization(object(), "context", "", "x")

    def test_unicode_cjk_tokenizer_splits_cjk_and_words(self):
        self.assertEqual(tokenize("你好, Paris!"), ["你", "好", ",", "paris", "!"])

    def test_rep_n_uses_one_minus_unique_over_total(self):
        self.assertAlmostEqual(rep_n_from_tokens("a b a b".split(), 2), 1 / 3)
        self.assertIsNone(rep_n_from_tokens(["a"], 2))

    def test_repetition_macro_reports_short_and_empty(self):
        report = repetition_report(
            ["a b a b", "", "x"],
            ns=(2,),
            tokenizer="unicode_cjk",
            short_policy="zero",
        )
        metric = report["metrics"]["rep_2"]
        self.assertAlmostEqual(metric["macro_average"], 1 / 9)
        self.assertEqual(metric["short_count"], 2)
        self.assertEqual(report["empty_count"], 1)

    def test_repetition_can_exclude_short_texts(self):
        report = repetition_report(
            ["a b a b", "x"],
            ns=(2,),
            tokenizer="unicode_cjk",
            short_policy="exclude",
        )
        metric = report["metrics"]["rep_2"]
        self.assertAlmostEqual(metric["macro_average"], 1 / 3)
        self.assertEqual(metric["included_count"], 1)

    def test_ppl_interval_summary(self):
        inside = summarize_ppl_against_human_interval(10.0, 8.0, 12.0)
        below = summarize_ppl_against_human_interval(6.0, 8.0, 12.0)
        above = summarize_ppl_against_human_interval(15.0, 8.0, 12.0)
        self.assertTrue(inside["inside_human_interval"])
        self.assertEqual(below["position"], "below")
        self.assertEqual(below["distance_to_human_interval"], 2.0)
        self.assertEqual(above["position"], "above")
        self.assertEqual(above["distance_to_human_interval"], 3.0)

    def test_ppl_interval_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            summarize_ppl_against_human_interval(10.0, 12.0, 8.0)

    def test_pairwise_position_swap_consistent_candidate_win(self):
        report = aggregate_pairwise_judgments(
            [judgment("1", forward_winner="A", reverse_winner="B")],
            **JUDGE_PROTOCOL,
            length_tokenizer="unicode_cjk",
            length_ratio_lower=0.8,
            length_ratio_upper=1.25,
        )
        self.assertEqual(report["overall"]["candidate_wins"], 1)
        self.assertEqual(report["overall"]["position_consistency_rate"], 1.0)

    def test_pairwise_position_disagreement_becomes_tie(self):
        report = aggregate_pairwise_judgments(
            [judgment("1", forward_winner="A", reverse_winner="A")],
            **JUDGE_PROTOCOL,
            length_tokenizer="unicode_cjk",
            length_ratio_lower=0.8,
            length_ratio_upper=1.25,
        )
        self.assertEqual(report["overall"]["ties"], 1)
        self.assertEqual(report["overall"]["candidate_tie_adjusted_win_rate"], 0.5)
        self.assertEqual(report["position_disagreement_tie_count"], 1)
        self.assertEqual(report["overall"]["position_consistency_rate"], 0.0)

    def test_pairwise_reports_length_groups(self):
        rows = [
            judgment(
                "short",
                forward_winner="A",
                reverse_winner="B",
                candidate_text="short",
                reference_text="one two three four",
            ),
            judgment(
                "long",
                forward_winner="B",
                reverse_winner="A",
                candidate_text="one two three four",
                reference_text="short",
            ),
        ]
        report = aggregate_pairwise_judgments(
            rows,
            **JUDGE_PROTOCOL,
            length_tokenizer="unicode_cjk",
            length_ratio_lower=0.8,
            length_ratio_upper=1.25,
        )
        self.assertEqual(report["by_length_group"]["shorter"]["count"], 1)
        self.assertEqual(report["by_length_group"]["longer"]["count"], 1)

    def test_length_controlled_win_rate_balances_length_strata(self):
        rows = [
            judgment(
                f"short-{index}",
                forward_winner="A",
                reverse_winner="B",
                candidate_text="short",
                reference_text="one two three four",
            )
            for index in range(3)
        ]
        rows.append(
            judgment(
                "long-loss",
                forward_winner="B",
                reverse_winner="A",
                candidate_text="one two three four",
                reference_text="short",
            )
        )
        report = aggregate_pairwise_judgments(
            rows,
            **JUDGE_PROTOCOL,
            length_tokenizer="unicode_cjk",
            length_ratio_lower=0.8,
            length_ratio_upper=1.25,
        )
        self.assertEqual(report["overall"]["candidate_tie_adjusted_win_rate"], 0.75)
        self.assertEqual(report["length_controlled_candidate_win_rate"], 0.5)

    def test_pairwise_requires_position_swap(self):
        row = judgment("1", forward_winner="A", reverse_winner="B")
        row["reverse"]["candidate_position"] = "A"
        with self.assertRaises(ValueError):
            aggregate_pairwise_judgments(
                [row],
                **JUDGE_PROTOCOL,
                length_tokenizer="unicode_cjk",
                length_ratio_lower=0.8,
                length_ratio_upper=1.25,
            )


if __name__ == "__main__":
    unittest.main()

