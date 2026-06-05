from __future__ import annotations

import json
from pathlib import Path

from jackbot.training.distill import _read_examples, target_output_paths


def test_distill_target_output_paths_keep_single_file_compatibility(tmp_path) -> None:
    output = tmp_path / "targets.jsonl"

    assert target_output_paths(output, 1) == [output]


def test_distill_target_output_paths_create_named_shards(tmp_path) -> None:
    output = tmp_path / "targets.jsonl"

    paths = target_output_paths(output, 3)

    assert paths == [
        tmp_path / "targets-0000.jsonl",
        tmp_path / "targets-0001.jsonl",
        tmp_path / "targets-0002.jsonl",
    ]


def test_read_examples_accepts_multiple_files_and_directories(tmp_path) -> None:
    first = tmp_path / "a.jsonl"
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    second = shard_dir / "b.jsonl"
    ignored = shard_dir / "notes.txt"
    first.write_text(json.dumps({"id": 1}) + "\n")
    second.write_text(json.dumps({"id": 2}) + "\n")
    ignored.write_text("not jsonl")

    examples = _read_examples([first, shard_dir])

    assert examples == [{"id": 1}, {"id": 2}]
