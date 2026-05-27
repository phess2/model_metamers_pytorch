import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_AUDIO_CLASSIFICATION = _load_module_from_file(
    "analysis_audio_classification_for_tests",
    _ROOT / "src" / "analysis" / "audio_classification.py",
)
_GENERATE_AUDIO_ADVERSARIALS = _load_module_from_file(
    "generate_audio_adversarials_for_tests",
    _ROOT / "scripts" / "generate_audio_adversarials.py",
)

aggregate_multilabel_topk = _AUDIO_CLASSIFICATION.aggregate_multilabel_topk
decode_audioset_labels = _AUDIO_CLASSIFICATION.decode_audioset_labels
summarize_multilabel_logits = _AUDIO_CLASSIFICATION.summarize_multilabel_logits


class AudioClassificationTests(unittest.TestCase):
    def test_decode_audioset_labels_supports_common_label_shapes(self):
        self.assertEqual(decode_audioset_labels({}), [])
        self.assertEqual(decode_audioset_labels({"labels": 4}), [4])
        self.assertEqual(decode_audioset_labels({"labels": [1, 2]}), [1, 2])
        self.assertEqual(
            decode_audioset_labels({"labels": torch.tensor([3, 5], dtype=torch.int64)}),
            [3, 5],
        )

    def test_multilabel_summary_and_aggregate_topk_metrics(self):
        summary_a = summarize_multilabel_logits(
            torch.tensor([[0.0, 4.0, 3.0, 0.1, -1.0, 2.0]], dtype=torch.float32),
            true_labels=[2, 5],
        )
        summary_b = summarize_multilabel_logits(
            torch.tensor([[3.5, 0.2, 0.1, 0.0, -1.0, -2.0]], dtype=torch.float32),
            true_labels=[0],
        )

        self.assertFalse(summary_a["top1_hit"])
        self.assertTrue(summary_a["top5_hit"])
        self.assertEqual(summary_b["predicted_label_idx"], 0)
        self.assertTrue(summary_b["top1_hit"])
        self.assertTrue(summary_b["top5_hit"])

        metrics = aggregate_multilabel_topk([summary_a, summary_b])
        self.assertAlmostEqual(metrics["top1_acc"], 0.5)
        self.assertAlmostEqual(metrics["top5_acc"], 1.0)
        self.assertEqual(metrics["num_samples"], 2)
        self.assertEqual(metrics["num_scored_samples"], 2)


class AudioExampleLoadingTests(unittest.TestCase):
    def test_selected_examples_keep_labels_for_decoding(self):
        fake_examples = [
            {"ytid": b"a", "labels": [1, 7]},
            {"ytid": b"b", "labels": torch.tensor([2], dtype=torch.int64)},
            {"ytid": b"c", "labels": [4, 9]},
        ]

        def _fake_example_loader(*_args, **_kwargs):
            return iter(fake_examples)

        with patch.object(
            _GENERATE_AUDIO_ADVERSARIALS,
            "example_loader",
            side_effect=_fake_example_loader,
        ):
            selected = _GENERATE_AUDIO_ADVERSARIALS._load_selected_examples(
                tfrecord_path=Path("/tmp/fake.tfrecords"),
                sample_indices=[0, 2],
            )

        self.assertEqual(sorted(selected.keys()), [0, 2])
        self.assertEqual(decode_audioset_labels(selected[0]), [1, 7])
        self.assertEqual(decode_audioset_labels(selected[2]), [4, 9])


if __name__ == "__main__":
    unittest.main()
