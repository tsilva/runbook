#!/usr/bin/env python3
"""Create a small unstyled SFT mix for later style rewriting.

The output is JSONL with chat `messages` plus metadata. Rows marked
`restyle: true` are the examples intended to have assistant responses
rewritten before training.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_ID = "HuggingFaceTB/smoltalk2"
CONFIG = "SFT"

DEFAULT_PLAN = [
    {
        "name": "capability_magpie",
        "split": "smoltalk_smollm3_smol_magpie_ultra_no_think",
        "count": 300,
        "restyle": False,
    },
    {
        "name": "capability_rewrite",
        "split": "smoltalk_smollm3_smol_rewrite_no_think",
        "count": 150,
        "restyle": False,
    },
    {
        "name": "capability_summarize",
        "split": "smoltalk_smollm3_smol_summarize_no_think",
        "count": 100,
        "restyle": False,
    },
    {
        "name": "capability_instruction_following",
        "split": "tulu_3_sft_personas_instruction_following_no_think",
        "count": 100,
        "restyle": False,
    },
    {
        "name": "capability_science",
        "split": "Mixture_of_Thoughts_science_no_think",
        "count": 50,
        "restyle": False,
    },
    {
        "name": "restyle_magpie",
        "split": "smoltalk_smollm3_smol_magpie_ultra_no_think",
        "count": 180,
        "restyle": True,
    },
    {
        "name": "restyle_rewrite",
        "split": "smoltalk_smollm3_smol_rewrite_no_think",
        "count": 70,
        "restyle": True,
    },
    {
        "name": "restyle_summarize",
        "split": "smoltalk_smollm3_smol_summarize_no_think",
        "count": 50,
        "restyle": True,
    },
]


def valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    has_user = False
    has_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            return False
        role = message.get("role")
        content = message.get("content")
        if role == "user" and isinstance(content, str) and content.strip():
            has_user = True
        if role == "assistant" and isinstance(content, str) and content.strip():
            has_assistant = True
    return has_user and has_assistant


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned = []
    for message in messages:
        role = str(message["role"])
        content = str(message["content"]).strip()
        cleaned.append({"role": role, "content": content})
    return cleaned


def stream_rows(split: str, seed: int, buffer_size: int) -> Iterator[dict[str, Any]]:
    dataset = load_dataset(DATASET_ID, CONFIG, split=split, streaming=True)
    shuffled = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    for row in shuffled:
        messages = row.get("messages")
        if valid_messages(messages):
            yield row


def collect(plan: list[dict[str, Any]], seed: int, buffer_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for item_index, item in enumerate(plan):
        split = item["split"]
        target_count = int(item["count"])
        collected = 0
        source_iter = stream_rows(split, seed + item_index, buffer_size)

        for row in source_iter:
            messages = clean_messages(row["messages"])
            key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            rows.append(
                {
                    "messages": messages,
                    "restyle": bool(item["restyle"]),
                    "bucket": item["name"],
                    "source_dataset": DATASET_ID,
                    "source_config": CONFIG,
                    "source_split": split,
                    "source": row.get("source", split),
                }
            )
            collected += 1
            if collected >= target_count:
                break

        if collected != target_count:
            raise RuntimeError(
                f"Only collected {collected} of {target_count} rows for {item['name']} ({split})"
            )

    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: Path, rows: list[dict[str, Any]], seed: int, buffer_size: int) -> None:
    counts: dict[str, int] = {}
    restyle_counts = {False: 0, True: 0}
    for row in rows:
        counts[row["bucket"]] = counts.get(row["bucket"], 0) + 1
        restyle_counts[bool(row["restyle"])] += 1

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_id": DATASET_ID,
        "config": CONFIG,
        "seed": seed,
        "buffer_size": buffer_size,
        "total_rows": len(rows),
        "preserve_rows": restyle_counts[False],
        "restyle_rows": restyle_counts[True],
        "buckets": counts,
        "notes": [
            "Rows with restyle=false should stay unchanged to preserve assistant behavior.",
            "Rows with restyle=true are untransformed placeholders for style rewriting.",
            "After rewriting, train on the same JSONL shape with the restyle metadata kept or dropped.",
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="datasets/style_mix_untransformed/train.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--manifest",
        default="datasets/style_mix_untransformed/manifest.json",
        help="Output manifest path.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect(DEFAULT_PLAN, seed=args.seed, buffer_size=args.buffer_size)
    write_jsonl(Path(args.output), rows)
    write_manifest(Path(args.manifest), rows, seed=args.seed, buffer_size=args.buffer_size)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
