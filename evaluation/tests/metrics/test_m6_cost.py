import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics.m6_cost import cost_record, cost_report, queries_to_target


def target_point(method, budget, value, metric="agreement_rate"):
    return {
        "method": method,
        "budget": budget,
        "value": value,
        "effect_family": "M2",
        "metric": metric,
        "unit": "rate_0_to_1",
    }


class M6MetricsTest(unittest.TestCase):
    def base_row(self):
        return {
            "id": "run-a",
            "stage": "attack",
            "currency": "USD",
            "price_as_of": "2026-07-13",
            "sample_count": 0,
            "data_cost": 0.0,
            "human_cost": 0.0,
            "other_cost": 0.0,
        }

    def test_per_query_and_gpu_costs(self):
        row = self.base_row() | {
            "api_billing": "per_query",
            "query_count": 500,
            "price_per_query": 0.02,
            "gpu_hours": 6,
            "gpu_price_per_hour": 2,
        }
        report = cost_record(row)
        money = report["monetary_cost"]
        self.assertEqual(money["api_cost"], 10.0)
        self.assertEqual(money["gpu_cost"], 12.0)
        self.assertEqual(money["known_subtotal"], 22.0)
        self.assertTrue(money["complete"])
        self.assertIn("samples", report["reported_cost_dimensions"])
        self.assertIn("money", report["reported_cost_dimensions"])

    def test_token_billing(self):
        row = self.base_row() | {
            "api_billing": "tokens",
            "input_tokens": 2_000_000,
            "output_tokens": 1_000_000,
            "input_price_per_million_tokens": 1.5,
            "output_price_per_million_tokens": 2.0,
            "gpu_hours": 0,
            "gpu_price_per_hour": 0,
        }
        self.assertEqual(cost_record(row)["monetary_cost"]["api_cost"], 5.0)

    def test_unknown_gpu_cost_stays_unknown(self):
        row = self.base_row() | {"api_billing": "none"}
        money = cost_record(row)["monetary_cost"]
        self.assertIsNone(money["gpu_cost"])
        self.assertFalse(money["complete"])
        self.assertIn("gpu_cost", money["unknown_components"])

    def test_mixed_api_billing_is_rejected(self):
        row = self.base_row() | {
            "api_billing": "per_query",
            "query_count": 10,
            "price_per_query": 1,
            "input_price_per_million_tokens": 2,
        }
        with self.assertRaises(ValueError):
            cost_record(row)

    def test_api_billing_must_be_explicit(self):
        with self.assertRaises(ValueError):
            cost_record(self.base_row())

    def test_price_date_must_be_explicit(self):
        row = self.base_row() | {"api_billing": "none"}
        del row["price_as_of"]
        with self.assertRaisesRegex(ValueError, "price_as_of"):
            cost_record(row)

    def test_at_least_two_cost_dimensions_are_required(self):
        row = self.base_row() | {"api_billing": "unknown"}
        row.pop("sample_count")
        row["data_cost"] = None
        row["human_cost"] = None
        row["other_cost"] = None
        with self.assertRaisesRegex(ValueError, "at least two cost dimensions"):
            cost_record(row)

    def test_unknown_api_cost_stays_unknown(self):
        row = self.base_row() | {
            "api_billing": "unknown",
            "query_count": 500,
            "gpu_hours": 0,
            "gpu_price_per_hour": 0,
        }
        money = cost_record(row)["monetary_cost"]
        self.assertIsNone(money["api_cost"])
        self.assertFalse(money["complete"])

    def test_none_billing_rejects_price_fields(self):
        row = self.base_row() | {
            "api_billing": "none",
            "price_per_query": 1.0,
        }
        with self.assertRaises(ValueError):
            cost_record(row)

    def test_count_fields_reject_fractions(self):
        row = self.base_row() | {
            "api_billing": "unknown",
            "query_count": 1.5,
        }
        with self.assertRaises(ValueError):
            cost_record(row)

    def test_duplicate_ids_are_rejected(self):
        row = self.base_row() | {"api_billing": "none"}
        with self.assertRaises(ValueError):
            cost_report([row, row])

    def test_queries_to_target_reports_censoring(self):
        points = [
            target_point("a", 100, 0.6),
            target_point("a", 500, 0.85),
            target_point("b", 100, 0.4),
            target_point("b", 500, 0.7),
        ]
        report = queries_to_target(points, target=0.8, budget_cap=500, higher_is_better=True)
        by_method = {row["method"]: row for row in report["results"]}
        self.assertEqual(by_method["a"]["queries_to_target"], 500)
        self.assertFalse(by_method["a"]["censored"])
        self.assertIsNone(by_method["b"]["queries_to_target"])
        self.assertTrue(by_method["b"]["censored"])
        self.assertEqual(by_method["b"]["display"], ">500")

    def test_queries_to_target_censors_at_last_observation(self):
        report = queries_to_target(
            [target_point("a", 100, 0.4)],
            target=0.8,
            budget_cap=500,
            higher_is_better=True,
        )
        result = report["results"][0]
        self.assertEqual(result["censor_at"], 100)
        self.assertEqual(result["display"], ">100")
        self.assertFalse(result["observed_at_budget_cap"])

    def test_queries_to_target_rejects_duplicate_method_budget(self):
        with self.assertRaises(ValueError):
            queries_to_target(
                [
                    target_point("a", 100, 0.4),
                    target_point("a", 100, 0.5),
                ],
                target=0.8,
                budget_cap=500,
                higher_is_better=True,
            )

    def test_queries_to_target_rejects_mixed_effect_metrics(self):
        with self.assertRaisesRegex(ValueError, "one effect_family"):
            queries_to_target(
                [
                    target_point("a", 100, 0.4),
                    target_point("b", 100, 12.0, metric="perplexity"),
                ],
                target=0.8,
                budget_cap=500,
                higher_is_better=True,
            )


if __name__ == "__main__":
    unittest.main()

