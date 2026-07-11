"""将 golden_set.json 同步到 LangSmith Dataset

Usage:
    docker exec shilian-app python eval/scripts/upload_golden_set.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("LANGSMITH_API_KEY not set. Skipping upload.")
        return

    golden_path = Path(__file__).resolve().parent.parent / "golden_set.json"
    if not golden_path.exists():
        print(f"Golden set not found: {golden_path}")
        return

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    entries = golden_data.get("entries", [])
    if not entries:
        print("No entries in golden set")
        return

    # 导入 langsmith SDK
    from langsmith import Client

    client = Client()
    dataset_name = "shilian-golden-set"

    # 如果 dataset 已存在，先删（重新创建保证一致）
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        client.delete_dataset(dataset_id=existing.id)
        print(f"Deleted existing dataset: {dataset_name}")
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=f"试炼 RAG 评估 golden set ({len(entries)} entries)",
    )

    for entry in entries:
        inputs = {"query": entry["query"]}
        outputs = {
            "relevant_chunk_ids": entry.get("relevant_chunk_ids", []),
            "reference_answer": entry.get("reference_answer", ""),
            "position_tag": entry.get("position_tag", ""),
            "difficulty": entry.get("difficulty", ""),
        }
        client.create_example(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset.id,
        )

    print(f"Uploaded {len(entries)} examples to LangSmith dataset '{dataset_name}'")


if __name__ == "__main__":
    main()
