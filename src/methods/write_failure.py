from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from swap.paper.scripts.analyze_write_failure import (
    compute_attention_metrics,
    compute_hidden_state_metrics,
    compute_norm_mismatch,
)


class WriteFailureRecorder:
    def __init__(
        self,
        *,
        method_name: str,
        dump_dir: str | None,
        max_samples: int,
        heatmap_samples: int,
        query_last_k: int,
        answer_source: str,
    ):
        self.method_name = str(method_name)
        self.max_samples = max(0, int(max_samples))
        self.heatmap_samples = max(0, int(heatmap_samples))
        self.query_last_k = max(1, int(query_last_k))
        self.answer_source = str(answer_source).strip().lower()

        if self.answer_source not in {"label", "normal_prediction", "steered_prediction"}:
            raise ValueError(
                "write-failure answer_source must be one of: label, normal_prediction, steered_prediction"
            )

        self.enabled = bool(dump_dir) and self.max_samples > 0
        self.dump_dir = Path(dump_dir).expanduser() if self.enabled and dump_dir else None
        self.raw_dir = (self.dump_dir / "raw") if self.dump_dir is not None else None
        if self.raw_dir is not None:
            self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.records: List[Dict[str, Any]] = []
        self._saved_raw = 0

    def should_record(self) -> bool:
        return self.enabled and len(self.records) < self.max_samples

    def choose_answer_text(
        self,
        sample: Dict[str, Any],
        *,
        normal_prediction: str,
        steered_prediction: str,
    ) -> str:
        if self.answer_source == "normal_prediction" and normal_prediction:
            return str(normal_prediction)
        if self.answer_source == "steered_prediction" and steered_prediction:
            return str(steered_prediction)
        return str(sample["label"])

    @staticmethod
    def infer_image_token_indices(inputs, processor=None) -> List[int]:
        if "mm_token_type_ids" in inputs:
            mm_token_type_ids = inputs["mm_token_type_ids"]
            if torch.is_tensor(mm_token_type_ids):
                mask = mm_token_type_ids[0].detach().to(device="cpu")
                return torch.nonzero(mask == 1, as_tuple=False).view(-1).tolist()

        if "input_ids" in inputs and processor is not None:
            token_ids = inputs["input_ids"][0].detach().to(device="cpu").tolist()
            tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
            return [
                idx
                for idx, token in enumerate(tokens)
                if "image_pad" in str(token).lower() or "image" in str(token).lower()
            ]

        return []

    @staticmethod
    def _stack_hidden_states(hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [layer.detach().to(device="cpu", dtype=torch.float32) for layer in hidden_states],
            dim=0,
        )

    @staticmethod
    def _stack_attention_rows(attentions: Sequence[torch.Tensor], query_indices: Sequence[int]) -> torch.Tensor:
        return torch.stack(
            [
                layer[:, :, list(query_indices), :].detach().to(device="cpu", dtype=torch.float32)
                for layer in attentions
            ],
            dim=0,
        )

    @staticmethod
    def _simplify_record(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question_id": record["question_id"],
            "label": record["label"],
            "normal_prediction": record["normal_prediction"],
            "steered_prediction": record["steered_prediction"],
            "normal_correct": record["normal_correct"],
            "steered_correct": record["steered_correct"],
            "analysis_answer_text": record["analysis_answer_text"],
            "visual_attention_ratio_normal": record["attention"]["visual_attention_ratio_normal"],
            "visual_attention_ratio_steered": record["attention"]["visual_attention_ratio_steered"],
            "visual_attention_ratio_drop_percent": record["attention"]["visual_attention_ratio_drop_percent"],
            "normalized_attention_entropy_normal": record["attention"]["normalized_attention_entropy_normal"],
            "normalized_attention_entropy_steered": record["attention"]["normalized_attention_entropy_steered"],
            "representation_cosine_similarity": record["hidden_state"]["representation_cosine_similarity"],
            "query_hidden_l2_ratio": record["hidden_state"]["query_hidden_l2_ratio"],
            "task_vector_to_hidden_norm_ratio": record["norm_mismatch"].get("task_vector_to_hidden_norm_ratio"),
            "raw_bundle_paths": record.get("raw_bundle_paths"),
        }

    def _save_raw_pair(
        self,
        *,
        sample_index: int,
        sample: Dict[str, Any],
        image_token_indices: Sequence[int],
        hidden_query_indices: Sequence[int],
        normal_hidden: torch.Tensor,
        steered_hidden: torch.Tensor,
        normal_attention_rows: torch.Tensor,
        steered_attention_rows: torch.Tensor,
        task_vector_obj: Any | None,
        answer_text: str,
        normal_raw_output: str,
        steered_raw_output: str,
    ) -> Dict[str, str] | None:
        if self.raw_dir is None or self._saved_raw >= self.heatmap_samples:
            return None

        stem = f"{sample_index:03d}_{self.method_name}_q{sample.get('question_id', sample_index)}"
        normal_path = self.raw_dir / f"{stem}.normal.pt"
        steered_path = self.raw_dir / f"{stem}.steered.pt"
        meta_path = self.raw_dir / f"{stem}.meta.json"

        normal_bundle = {
            "hidden_states": normal_hidden.to(dtype=torch.float16),
            "attentions": normal_attention_rows.to(dtype=torch.float16),
            "image_token_indices": list(image_token_indices),
            "query_token_indices": list(hidden_query_indices),
            "answer_text": answer_text,
            "raw_output": normal_raw_output,
            "question_id": sample.get("question_id"),
        }
        steered_bundle = {
            "hidden_states": steered_hidden.to(dtype=torch.float16),
            "attentions": steered_attention_rows.to(dtype=torch.float16),
            "image_token_indices": list(image_token_indices),
            "query_token_indices": list(hidden_query_indices),
            "answer_text": answer_text,
            "raw_output": steered_raw_output,
            "question_id": sample.get("question_id"),
            "task_vector": task_vector_obj,
        }

        torch.save(normal_bundle, normal_path)
        torch.save(steered_bundle, steered_path)
        meta_payload = {
            "method": self.method_name,
            "question_id": sample.get("question_id"),
            "label": str(sample.get("label", "")),
            "analysis_answer_text": answer_text,
            "image": sample.get("image"),
            "normal_bundle": str(normal_path),
            "steered_bundle": str(steered_path),
            "query_token_indices": list(hidden_query_indices),
            "image_token_indices": list(image_token_indices),
        }
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._saved_raw += 1
        return {
            "normal": str(normal_path),
            "steered": str(steered_path),
            "meta": str(meta_path),
        }

    def record_pair(
        self,
        *,
        sample: Dict[str, Any],
        sample_index: int,
        full_inputs,
        prompt_len: int,
        normal_outputs,
        steered_outputs,
        task_vector_obj: Any | None,
        analysis_answer_text: str,
        normal_raw_output: str,
        steered_raw_output: str,
        normal_prediction: str,
        steered_prediction: str,
        processor=None,
    ) -> None:
        if not self.should_record():
            return

        image_token_indices = self.infer_image_token_indices(full_inputs, processor=processor)
        if not image_token_indices:
            return

        full_seq_len = int(full_inputs["input_ids"].shape[-1])
        answer_token_indices = list(range(int(prompt_len), full_seq_len))
        if not answer_token_indices:
            return

        hidden_query_indices = answer_token_indices[-min(self.query_last_k, len(answer_token_indices)) :]
        attention_query_indices = hidden_query_indices

        normal_hidden = self._stack_hidden_states(normal_outputs.hidden_states)
        steered_hidden = self._stack_hidden_states(steered_outputs.hidden_states)
        normal_attention_rows = self._stack_attention_rows(normal_outputs.attentions, attention_query_indices)
        steered_attention_rows = self._stack_attention_rows(steered_outputs.attentions, attention_query_indices)

        seq_len = int(normal_hidden.shape[-2])
        image_mask = torch.zeros(seq_len, dtype=torch.bool)
        for token_idx in image_token_indices:
            if 0 <= int(token_idx) < seq_len:
                image_mask[int(token_idx)] = True

        if not image_mask.any():
            return

        attention_metrics = compute_attention_metrics(
            normal_attention_rows,
            steered_attention_rows,
            image_mask=image_mask[: int(normal_attention_rows.shape[-1])],
            query_indices=list(range(len(hidden_query_indices))),
        )
        hidden_metrics = compute_hidden_state_metrics(
            normal_hidden,
            steered_hidden,
            image_mask=image_mask,
            query_indices=hidden_query_indices,
        )
        norm_mismatch = compute_norm_mismatch(
            task_vector_obj,
            normal_hidden,
            image_mask=image_mask,
        )

        raw_bundle_paths = self._save_raw_pair(
            sample_index=sample_index,
            sample=sample,
            image_token_indices=image_token_indices,
            hidden_query_indices=hidden_query_indices,
            normal_hidden=normal_hidden,
            steered_hidden=steered_hidden,
            normal_attention_rows=normal_attention_rows,
            steered_attention_rows=steered_attention_rows,
            task_vector_obj=task_vector_obj,
            answer_text=analysis_answer_text,
            normal_raw_output=normal_raw_output,
            steered_raw_output=steered_raw_output,
        )

        record = {
            "sample_index": int(sample_index),
            "question_id": sample.get("question_id"),
            "image": sample.get("image"),
            "label": str(sample.get("label", "")),
            "analysis_answer_text": analysis_answer_text,
            "normal_raw_output": normal_raw_output,
            "steered_raw_output": steered_raw_output,
            "normal_prediction": normal_prediction,
            "steered_prediction": steered_prediction,
            "normal_correct": str(normal_prediction) == str(sample.get("label", "")),
            "steered_correct": str(steered_prediction) == str(sample.get("label", "")),
            "prompt_len": int(prompt_len),
            "full_seq_len": full_seq_len,
            "query_token_indices": list(hidden_query_indices),
            "image_token_indices": list(image_token_indices),
            "attention": attention_metrics,
            "hidden_state": hidden_metrics,
            "norm_mismatch": norm_mismatch,
            "raw_bundle_paths": raw_bundle_paths,
        }
        self.records.append(record)

    @staticmethod
    def _mean(values: List[float | None]) -> float | None:
        filtered = [float(value) for value in values if value is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    def export(self) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "num_analyzed_samples": 0,
            }

        summary_rows = [self._simplify_record(record) for record in self.records]
        export = {
            "enabled": True,
            "method": self.method_name,
            "dump_dir": str(self.dump_dir) if self.dump_dir is not None else None,
            "raw_dir": str(self.raw_dir) if self.raw_dir is not None else None,
            "num_analyzed_samples": len(self.records),
            "query_last_k": self.query_last_k,
            "answer_source": self.answer_source,
            "summary": {
                "visual_attention_ratio_normal": self._mean(
                    [row["visual_attention_ratio_normal"] for row in summary_rows]
                ),
                "visual_attention_ratio_steered": self._mean(
                    [row["visual_attention_ratio_steered"] for row in summary_rows]
                ),
                "visual_attention_ratio_drop_percent": self._mean(
                    [row["visual_attention_ratio_drop_percent"] for row in summary_rows]
                ),
                "normalized_attention_entropy_normal": self._mean(
                    [row["normalized_attention_entropy_normal"] for row in summary_rows]
                ),
                "normalized_attention_entropy_steered": self._mean(
                    [row["normalized_attention_entropy_steered"] for row in summary_rows]
                ),
                "representation_cosine_similarity": self._mean(
                    [row["representation_cosine_similarity"] for row in summary_rows]
                ),
                "query_hidden_l2_ratio": self._mean(
                    [row["query_hidden_l2_ratio"] for row in summary_rows]
                ),
                "task_vector_to_hidden_norm_ratio": self._mean(
                    [row["task_vector_to_hidden_norm_ratio"] for row in summary_rows]
                ),
                "normal_accuracy": self._mean([1.0 if row["normal_correct"] else 0.0 for row in summary_rows]),
                "steered_accuracy": self._mean([1.0 if row["steered_correct"] else 0.0 for row in summary_rows]),
            },
            "samples": summary_rows,
            "raw_samples": [row for row in summary_rows if row.get("raw_bundle_paths")],
        }
        return export
