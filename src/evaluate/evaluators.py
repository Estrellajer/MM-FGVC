from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class EvaluatorBase:
    name: str

    def evaluate(
        self,
        predictions: Sequence[str],
        labels: Sequence[str],
        samples: Sequence[dict],
    ) -> Dict[str, float]:
        raise NotImplementedError


class RawAccuracyEvaluator(EvaluatorBase):
    def __init__(self):
        super().__init__(name="raw_accuracy")

    def evaluate(
        self,
        predictions: Sequence[str],
        labels: Sequence[str],
        samples: Sequence[dict],
    ) -> Dict[str, float]:
        if len(predictions) != len(labels):
            raise ValueError("predictions and labels length mismatch")
        total = len(labels)
        correct = sum(pred == label for pred, label in zip(predictions, labels))
        return {
            "metric": self.name,
            "accuracy": (correct / total) if total > 0 else 0.0,
            "correct": float(correct),
            "total": float(total),
        }


class PairAccuracyEvaluator(EvaluatorBase):
    def __init__(self):
        super().__init__(name="pair_accuracy")

    def evaluate(
        self,
        predictions: Sequence[str],
        labels: Sequence[str],
        samples: Sequence[dict],
    ) -> Dict[str, float]:
        if len(predictions) != len(labels):
            raise ValueError("predictions and labels length mismatch")
        if len(samples) % 2 != 0:
            raise ValueError("Pair evaluator expects number of samples to be even")

        pair_total = len(samples) // 2
        pair_correct = 0
        raw_correct = 0

        for i in range(0, len(samples), 2):
            first_ok = predictions[i] == labels[i]
            second_ok = predictions[i + 1] == labels[i + 1]
            raw_correct += int(first_ok) + int(second_ok)
            if first_ok and second_ok:
                pair_correct += 1

        raw_total = len(samples)
        return {
            "metric": self.name,
            "pair_accuracy": (pair_correct / pair_total) if pair_total > 0 else 0.0,
            "pair_correct": float(pair_correct),
            "pair_total": float(pair_total),
            "raw_accuracy": (raw_correct / raw_total) if raw_total > 0 else 0.0,
        }


class NaturalBenchGroupEvaluator(EvaluatorBase):
    def __init__(self):
        super().__init__(name="naturalbench_group")

    def evaluate(
        self,
        predictions: Sequence[str],
        labels: Sequence[str],
        samples: Sequence[dict],
    ) -> Dict[str, float]:
        if len(predictions) != len(labels):
            raise ValueError("predictions and labels length mismatch")
        total_samples = len(samples)
        usable_samples = total_samples - (total_samples % 4)
        dropped_samples = total_samples - usable_samples

        if usable_samples == 0:
            raw_correct_all = sum(pred == label for pred, label in zip(predictions, labels))
            raw_acc_all = (raw_correct_all / total_samples) if total_samples > 0 else 0.0
            return {
                "metric": self.name,
                "q_acc": 0.0,
                "i_acc": 0.0,
                "g_acc": 0.0,
                "raw_acc": raw_acc_all,
                "group_total": 0.0,
                "used_samples": 0.0,
                "dropped_samples": float(dropped_samples),
            }

        total_groups = usable_samples // 4
        q_correct = 0
        i_correct = 0
        g_correct = 0
        raw_correct = 0

        for i in range(0, usable_samples, 4):
            group_hits = [predictions[j] == labels[j] for j in range(i, i + 4)]
            raw_correct += sum(group_hits)

            if group_hits[0] and group_hits[1]:
                q_correct += 1
            if group_hits[2] and group_hits[3]:
                q_correct += 1

            if group_hits[0] and group_hits[2]:
                i_correct += 1
            if group_hits[1] and group_hits[3]:
                i_correct += 1

            if all(group_hits):
                g_correct += 1

        q_total = total_groups * 2
        i_total = total_groups * 2
        raw_total = total_groups * 4
        return {
            "metric": self.name,
            "q_acc": (q_correct / q_total) if q_total > 0 else 0.0,
            "i_acc": (i_correct / i_total) if i_total > 0 else 0.0,
            "g_acc": (g_correct / total_groups) if total_groups > 0 else 0.0,
            "raw_acc": (raw_correct / raw_total) if raw_total > 0 else 0.0,
            "group_total": float(total_groups),
            "used_samples": float(usable_samples),
            "dropped_samples": float(dropped_samples),
        }


EVALUATOR_REGISTRY = {
    "raw": RawAccuracyEvaluator,
    "pair": PairAccuracyEvaluator,
    "naturalbench_group": NaturalBenchGroupEvaluator,
}


def build_evaluator(evaluator_name: str) -> EvaluatorBase:
    if evaluator_name not in EVALUATOR_REGISTRY:
        supported = ", ".join(sorted(EVALUATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown evaluator '{evaluator_name}'. Supported evaluators: {supported}"
        )
    return EVALUATOR_REGISTRY[evaluator_name]()
