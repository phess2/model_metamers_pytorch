"""Select AudioSet example indices with at least N labels."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

try:
    from tfrecord.reader import example_loader
except ModuleNotFoundError:
    example_loader = None

from src.analysis.audio_classification import decode_audioset_labels


def main() -> None:
    parser = ArgumentParser(description="Select AudioSet examples with >=N labels.")
    parser.add_argument(
        "--audioset_root",
        type=str,
        default="/home/rphess/orcd/datasets/AudioSet",
    )
    parser.add_argument("--audioset_split_glob", type=str, default="audioset-2m-part*")
    parser.add_argument("--tfrecord_glob", type=str, default="*.tfrecords")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--min_labels", type=int, default=3)
    parser.add_argument(
        "--max_scan_examples",
        type=int,
        default=20_000,
        help="Stop after scanning this many TFRecord examples.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="JSON output path for selected index/label manifest.",
    )
    args = parser.parse_args()

    if example_loader is None:
        raise ModuleNotFoundError(
            "Missing dependency 'tfrecord'. Install it to load AudioSet TFRecord examples."
        )
    if args.num_examples <= 0:
        raise SystemExit("--num_examples must be positive.")
    if args.min_labels <= 0:
        raise SystemExit("--min_labels must be positive.")

    audioset_root = Path(args.audioset_root)
    split_dirs = sorted(audioset_root.glob(args.audioset_split_glob))
    if not split_dirs:
        raise SystemExit(
            f"No split dirs matched {args.audioset_split_glob!r} under {audioset_root}"
        )
    first_split_dir = split_dirs[0]
    tfrecord_paths = sorted(first_split_dir.glob(args.tfrecord_glob))
    if not tfrecord_paths:
        raise SystemExit(
            f"No TFRecords matched {args.tfrecord_glob!r} under {first_split_dir}"
        )
    tfrecord_path = tfrecord_paths[0]

    selected: list[dict[str, object]] = []
    examples = example_loader(
        str(tfrecord_path),
        index_path=None,
        description=None,
        compression_type="gzip",
    )
    for example_idx, example in enumerate(examples):
        if example_idx >= args.max_scan_examples:
            break
        labels = decode_audioset_labels(example)
        if len(labels) < args.min_labels:
            continue
        ytid = example.get("ytid", b"")
        if isinstance(ytid, bytes):
            ytid = ytid.decode("utf-8", errors="replace")
        selected.append(
            {
                "sample_idx": int(example_idx),
                "ytid": str(ytid),
                "labels": [int(x) for x in labels],
                "num_labels": int(len(labels)),
            }
        )
        if len(selected) >= args.num_examples:
            break

    if len(selected) < args.num_examples:
        raise SystemExit(
            f"Only found {len(selected)} examples with >= {args.min_labels} labels "
            f"after scanning {args.max_scan_examples} examples."
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audioset_root": str(audioset_root),
        "first_split_dir": str(first_split_dir),
        "tfrecord_path": str(tfrecord_path),
        "num_examples_requested": int(args.num_examples),
        "min_labels": int(args.min_labels),
        "selected_examples": selected,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"Saved {len(selected)} selected examples to {output_path}")
    print("Indices:", ",".join(str(item["sample_idx"]) for item in selected))


if __name__ == "__main__":
    main()
