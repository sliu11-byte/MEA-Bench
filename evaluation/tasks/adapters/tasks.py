from __future__ import annotations

import re
from typing import Any

from evaluation.core.config import DatasetSpec
from evaluation.core.choice_scoring import with_leading_space
from evaluation.tasks.parsing import CodeCompletionParser, LabelParser, MultipleChoiceParser, NumericParser, normalize_number
from .base import AdaptedExample, DatasetAdapter, format_multiple_choice


LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _base(adapter: DatasetAdapter, raw: dict[str, Any], split: str, index: int, **kwargs: Any) -> AdaptedExample:
    return AdaptedExample(
        example_id=adapter.make_example_id(raw, split, index),
        example_index=index,
        split=split,
        raw_example=raw,
        extra=kwargs.pop("extra", {}),
        **kwargs,
    )


class ArcChallengeAdapter(DatasetAdapter):
    prompt_template_id = "arc_challenge_zero_shot_mc"
    prompt_template_version = "2"
    label_space = list("ABCDE")

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        choices = raw.get("choices") or {}
        labels = [str(x).upper() for x in choices.get("label", [])]
        texts = [_text(x) for x in choices.get("text", [])]
        if not labels or len(labels) != len(texts):
            raise ValueError("ARC example has invalid choices")
        logical, rendered = format_multiple_choice(_text(raw.get("question")), list(zip(labels, texts)))
        gold = _text(raw.get("answerKey")).upper()
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=gold, normalized_gold_answer=gold,
                     parser=MultipleChoiceParser(labels), extra={
                         "choice_scoring": {
                             "method": "conditional_loglikelihood_v2",
                             "prompt": f"Question: {_text(raw.get('question')).strip()}\nAnswer:",
                             "labels": labels,
                             "continuations": with_leading_space(texts),
                             "length_normalize": True,
                             "metric": "acc_norm",
                         }
                     })


def _hellaswag_context(raw: dict[str, Any]) -> str:
    ctx = _text(raw.get("ctx"))
    if not ctx:
        ctx_a = _text(raw.get("ctx_a"))
        ctx_b = _text(raw.get("ctx_b"))
        ctx = f"{ctx_a} {ctx_b[:1].upper() + ctx_b[1:] if ctx_b else ''}".strip()
    return re.sub(r"\s*\[title\]\s*", ". ", ctx).strip()


class HellaSwagAdapter(DatasetAdapter):
    prompt_template_id = "hellaswag_zero_shot_completion_mc"
    prompt_template_version = "2"
    preprocessing_rule = "Remove [title] markers and normalize context whitespace."
    label_space = list("ABCD")

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        endings = [_text(x) for x in raw.get("endings", [])]
        labels = LETTERS[:len(endings)]
        context = _hellaswag_context(raw)
        logical, rendered = format_multiple_choice(
            f"Complete the following scenario:\n{context}",
            list(zip(labels, endings)),
        )
        answer_index = int(raw.get("label"))
        gold = labels[answer_index]
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=gold, normalized_gold_answer=gold,
                     parser=MultipleChoiceParser(labels), extra={
                         "choice_scoring": {
                             "method": "conditional_loglikelihood_v2",
                             "prompt": context,
                             "labels": labels,
                             "continuations": with_leading_space(endings),
                             "length_normalize": True,
                             "metric": "acc_norm",
                         }
                     })


class MMLUAdapter(DatasetAdapter):
    prompt_template_id = "mmlu_zero_shot_mc"
    prompt_template_version = "2"
    label_space = list("ABCD")

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        choices = [_text(x) for x in raw.get("choices", [])]
        labels = LETTERS[:len(choices)]
        subject = _text(raw.get("subject")).replace("_", " ")
        question = _text(raw.get("question"))
        if subject:
            question = f"Subject: {subject}\n{question}"
        logical, rendered = format_multiple_choice(question, list(zip(labels, choices)))
        gold = labels[int(raw.get("answer"))]
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=gold, normalized_gold_answer=gold,
                     parser=MultipleChoiceParser(labels), extra={
                         "choice_scoring": {
                             "method": "conditional_loglikelihood_v2",
                             "prompt": f"{logical}\nAnswer:",
                             "labels": labels,
                             "continuations": with_leading_space(labels),
                             "length_normalize": False,
                             "metric": "acc",
                         }
                     })


class TruthfulQAAdapter(DatasetAdapter):
    prompt_template_id = "truthfulqa_mc1_zero_shot"
    prompt_template_version = "2"

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        targets = raw.get("mc1_targets") or {}
        choices = [_text(x) for x in targets.get("choices", [])]
        target_labels = list(targets.get("labels", []))
        labels = LETTERS[:len(choices)]
        positives = [i for i, value in enumerate(target_labels) if int(value) == 1]
        if len(positives) != 1:
            raise ValueError(f"TruthfulQA mc1 expected one gold answer, found {positives}")
        logical, rendered = format_multiple_choice(_text(raw.get("question")), list(zip(labels, choices)))
        gold = labels[positives[0]]
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=gold, normalized_gold_answer=gold,
                     parser=MultipleChoiceParser(labels), extra={
                         "choice_scoring": {
                             "method": "conditional_loglikelihood_v2",
                             "prompt": f"Question: {_text(raw.get('question')).strip()}\nAnswer:",
                             "labels": labels,
                             "continuations": with_leading_space(choices),
                             "length_normalize": False,
                             "metric": "mc1",
                         }
                     })


class WinoGrandeAdapter(DatasetAdapter):
    prompt_template_id = "winogrande_zero_shot_mc"
    prompt_template_version = "2"
    label_space = ["A", "B"]
    label_mapping = {"1": "A", "2": "B"}

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        sentence = _text(raw.get("sentence"))
        option_texts = [_text(raw.get("option1")), _text(raw.get("option2"))]
        options = [("A", option_texts[0]), ("B", option_texts[1])]
        logical, rendered = format_multiple_choice(
            f"Choose the option that correctly replaces the blank in: {sentence}", options,
        )
        gold = self.label_mapping[_text(raw.get("answer"))]
        if sentence.count("_") != 1:
            raise ValueError("WinoGrande example must contain exactly one blank")
        prefix, suffix = sentence.split("_", 1)
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=gold, normalized_gold_answer=gold,
                     parser=MultipleChoiceParser(["A", "B"]), extra={
                         "choice_scoring": {
                             "method": "conditional_loglikelihood_v2",
                             "prompt": prefix,
                             "labels": ["A", "B"],
                             "continuations": [f"{text}{suffix}" for text in option_texts],
                             "length_normalize": False,
                             "metric": "acc",
                         }
                     })


class GSM8KAdapter(DatasetAdapter):
    prompt_template_id = "gsm8k_zero_shot_reasoning"
    prompt_template_version = "1"

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        question = _text(raw.get("question"))
        logical = f"Math word problem: {question}"
        rendered = (
            f"{logical}\n\nSolve the problem. You may show reasoning, but end with "
            "'Final answer: <NUMBER>'."
        )
        raw_gold = _text(raw.get("answer"))
        match = re.search(r"####\s*([^\n]+)", raw_gold)
        gold_value = match.group(1) if match else raw_gold
        normalized = normalize_number(gold_value)
        if normalized is None:
            raise ValueError(f"Could not normalize GSM8K gold answer: {raw_gold!r}")
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=gold_value.strip(), normalized_gold_answer=normalized,
                     parser=NumericParser())


def _medqa_options(raw_options: Any) -> list[tuple[str, str]]:
    if isinstance(raw_options, dict):
        return [(str(key).upper(), _text(value)) for key, value in raw_options.items()]
    result = []
    for position, option in enumerate(raw_options or []):
        if isinstance(option, dict):
            result.append((_text(option.get("key") or LETTERS[position]).upper(), _text(option.get("value"))))
        else:
            result.append((LETTERS[position], _text(option)))
    return result


class MedQAAdapter(DatasetAdapter):
    prompt_template_id = "paper_author_medqa_zero_shot_mc"
    prompt_template_version = "1"
    label_space = list("ABCDE")

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        options = _medqa_options(raw.get("options"))
        logical, rendered = format_multiple_choice(
            f"Medical question: {_text(raw.get('question'))}", options,
        )
        gold = _text(raw.get("answer_idx")).upper()
        labels = [label for label, _ in options]
        if gold not in labels:
            raise ValueError(f"MedQA gold {gold!r} not in labels {labels}")
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=gold, normalized_gold_answer=gold,
                     parser=MultipleChoiceParser(labels))


class PubMedQAAdapter(DatasetAdapter):
    prompt_template_id = "paper_author_pubmedqa_context_ynm"
    prompt_template_version = "1"
    label_space = ["yes", "no", "maybe"]
    label_mapping = {"yes": "yes", "no": "no", "maybe": "maybe"}

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        contexts = raw.get("CONTEXTS") or raw.get("contexts") or []
        if isinstance(contexts, dict):
            contexts = contexts.get("contexts") or contexts.get("text") or list(contexts.values())
        context_block = "\n\n".join(_text(item) for item in contexts)
        question = _text(raw.get("QUESTION") or raw.get("question"))
        logical = f"Question: {question}\n\nContexts:\n{context_block}"
        rendered = f"{logical}\n\nAnswer with exactly one label: yes, no, or maybe.\nFinal answer:"
        gold = _text(raw.get("final_decision") or raw.get("answer")).lower()
        parser = LabelParser(self.label_space)
        normalized = parser.normalize(gold)
        if normalized is None:
            raise ValueError(f"Unknown PubMedQA label: {gold!r}")
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=gold, normalized_gold_answer=normalized, parser=parser)


CHEMPROT_LABELS = [
    "ACTIVATOR", "INHIBITOR", "AGONIST", "ANTAGONIST", "SUBSTRATE", "PRODUCT-OF",
    "SUBSTRATE_PRODUCT-OF", "DIRECT-REGULATOR", "INDIRECT-DOWNREGULATOR",
    "INDIRECT-UPREGULATOR", "UPREGULATOR", "DOWNREGULATOR", "AGONIST-ACTIVATOR",
    "AGONIST-INHIBITOR",
]


class ChemProtAdapter(DatasetAdapter):
    prompt_template_id = "paper_author_chemprot_14way_mc"
    prompt_template_version = "1"
    label_space = CHEMPROT_LABELS
    label_mapping = {label: LETTERS[i] for i, label in enumerate(CHEMPROT_LABELS)}

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        options = [(LETTERS[i], label) for i, label in enumerate(CHEMPROT_LABELS)]
        logical, rendered = format_multiple_choice(
            f"Choose the best chemical-protein relation for this text:\n{_text(raw.get('text')).replace(chr(10), ' ')}",
            options,
        )
        raw_label = _text(raw.get("label")).upper()
        if raw_label not in self.label_mapping:
            raise ValueError(f"Unknown ChemProt label: {raw_label!r}")
        gold = self.label_mapping[raw_label]
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=raw_label, normalized_gold_answer=gold,
                     parser=MultipleChoiceParser([label for label, _ in options]))


class FOMCAdapter(DatasetAdapter):
    prompt_template_id = "paper_author_fomc_stance"
    prompt_template_version = "1"
    label_space = ["dovish", "hawkish", "neutral"]

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        instruction = _text(raw.get("instruction"))
        statement = _text(raw.get("text"))
        logical = f"{instruction}\n\nStatement: {statement}".strip()
        rendered = f"{logical}\n\nChoose one stance: dovish, hawkish, or neutral.\nFinal answer:"
        parser = LabelParser(self.label_space)
        gold = _text(raw.get("label")).lower()
        normalized = parser.normalize(gold)
        if normalized is None:
            raise ValueError(f"Unknown FOMC label: {gold!r}")
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=gold, normalized_gold_answer=normalized, parser=parser)


class HeadLineAdapter(DatasetAdapter):
    prompt_template_id = "paper_author_headline_price_sentiment"
    prompt_template_version = "1"
    label_space = ["negative", "neutral", "positive", "none"]

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        headline = _text(raw.get("News") or raw.get("headline"))
        logical = f"Financial headline: {headline}"
        rendered = (
            f"{logical}\n\nWhat is the price sentiment? Choose one: negative, neutral, positive, or none."
            "\nFinal answer:"
        )
        parser = LabelParser(self.label_space)
        gold = _text(raw.get("Price Sentiment") or raw.get("label")).lower()
        normalized = parser.normalize(gold)
        if normalized is None:
            raise ValueError(f"Unknown HeadLine label: {gold!r}")
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=gold, normalized_gold_answer=normalized, parser=parser)


class FPBAdapter(DatasetAdapter):
    prompt_template_id = "paper_author_fpb_sentiment"
    prompt_template_version = "1"
    label_space = ["negative", "neutral", "positive"]
    label_mapping = {"0": "negative", "1": "neutral", "2": "positive"}

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        sentence = _text(raw.get("sentence"))
        logical = f"Financial phrase: {sentence}"
        rendered = f"{logical}\n\nClassify the sentiment as negative, neutral, or positive.\nFinal answer:"
        raw_label = raw.get("label")
        gold = self.label_mapping.get(str(raw_label), _text(raw_label).lower())
        parser = LabelParser(self.label_space)
        normalized = parser.normalize(gold)
        if normalized is None:
            raise ValueError(f"Unknown FPB label: {raw_label!r}")
        return _base(self, raw, split, index, logical_prompt=logical, rendered_prompt=rendered,
                     gold_answer=gold, normalized_gold_answer=normalized, parser=parser)


class HumanEvalAdapter(DatasetAdapter):
    prompt_template_id = "official_humaneval_prompt"
    prompt_template_version = "1"
    preprocessing_rule = "Preserve the official problem prompt and function signature verbatim."

    def adapt(self, raw: dict[str, Any], split: str, index: int) -> AdaptedExample:
        prompt = _text(raw.get("prompt"))
        if not prompt:
            raise ValueError("HumanEval example is missing official prompt")
        return _base(
            self, raw, split, index,
            logical_prompt=prompt,
            rendered_prompt=prompt,
            gold_answer="",
            normalized_gold_answer=None,
            parser=CodeCompletionParser(),
            extra={
                "task_id": _text(raw.get("task_id")),
                "canonical_prompt": prompt,
                "canonical_solution": _text(raw.get("canonical_solution")),
                "test": _text(raw.get("test")),
                "entry_point": _text(raw.get("entry_point")),
            },
        )


ADAPTERS = {
    "arc_challenge": ArcChallengeAdapter,
    "hellaswag": HellaSwagAdapter,
    "mmlu": MMLUAdapter,
    "truthfulqa": TruthfulQAAdapter,
    "winogrande": WinoGrandeAdapter,
    "gsm8k": GSM8KAdapter,
    "medqa": MedQAAdapter,
    "pubmedqa": PubMedQAAdapter,
    "chemprot": ChemProtAdapter,
    "fomc": FOMCAdapter,
    "headline": HeadLineAdapter,
    "fpb": FPBAdapter,
    "humaneval": HumanEvalAdapter,
}


def create_adapter(spec: DatasetSpec) -> DatasetAdapter:
    try:
        adapter_class = ADAPTERS[spec.adapter]
    except KeyError as exc:
        raise KeyError(f"No M1 adapter registered for {spec.adapter!r}") from exc
    return adapter_class(spec)
