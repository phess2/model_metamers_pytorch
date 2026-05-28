"""Build a persistent AudioSet<->BEATs label-index mapping CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.request import urlopen

import torch

_DEFAULT_BEATS_CHECKPOINT = (
    "/orcd/data/jhm/001/om2/rphess/pretrained_models/beats/checkpoints/"
    "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt"
)
_DEFAULT_AUDIOSET_CLASS_LABELS = (
    "https://raw.githubusercontent.com/IBM/audioset-classification/master/"
    "audioset_classify/metadata/class_labels_indices.csv"
)
_DEFAULT_OUTPUT_CSV = (
    "src/analysis/metadata/beats_iter3_plus_as2m_audioset_label_map.csv"
)


def _load_audioset_labels(labels_source: str) -> dict[str, int]:
    if labels_source.startswith(("http://", "https://")):
        with urlopen(labels_source, timeout=15) as response:
            lines = response.read().decode("utf-8", errors="replace").splitlines()
    else:
        with Path(labels_source).open("r", encoding="utf-8", newline="") as handle:
            lines = handle.read().splitlines()
    reader = csv.DictReader(lines)
    mid_to_index: dict[str, int] = {}
    for row in reader:
        mid = row.get("mid")
        index = row.get("index")
        if mid is None or index is None:
            continue
        mid_to_index[mid] = int(index)
    return mid_to_index


def _load_beats_label_dict(checkpoint_path: str) -> dict[int, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_dict = checkpoint.get("label_dict")
    if not isinstance(label_dict, dict):
        raise RuntimeError(
            "Checkpoint does not include a dict label_dict. "
            "Use a fine-tuned BEATs checkpoint with classifier head."
        )
    ordered_label_dict = {int(k): str(v) for k, v in sorted(label_dict.items())}
    return ordered_label_dict


def build_mapping_rows(
    beats_index_to_mid: dict[int, str],
    audioset_mid_to_index: dict[str, int],
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for model_index, mid in beats_index_to_mid.items():
        audioset_index = audioset_mid_to_index.get(mid)
        if audioset_index is None:
            continue
        rows.append(
            {
                "audioset_index": int(audioset_index),
                "audioset_mid": mid,
                "model_index": int(model_index),
                "model_mid": mid,
            }
        )
    rows.sort(key=lambda row: int(row["audioset_index"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build persistent AudioSet<->BEATs label index mapping CSV."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=_DEFAULT_BEATS_CHECKPOINT,
    )
    parser.add_argument(
        "--audioset_labels_source",
        type=str,
        default=_DEFAULT_AUDIOSET_CLASS_LABELS,
        help="Path or URL to AudioSet class_labels_indices.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=_DEFAULT_OUTPUT_CSV,
    )
    args = parser.parse_args()

    beats_index_to_mid = _load_beats_label_dict(args.checkpoint_path)
    audioset_mid_to_index = _load_audioset_labels(args.audioset_labels_source)
    rows = build_mapping_rows(beats_index_to_mid, audioset_mid_to_index)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audioset_index", "audioset_mid", "model_index", "model_mid"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} label mappings to {output_path} "
        f"from checkpoint {args.checkpoint_path}."
    )


if __name__ == "__main__":
    main()
