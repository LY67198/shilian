"""RAGAS 5-metric evaluation for Phase 3 retrieval pipeline.

Usage:
    docker exec shilian-app python eval/scripts/eval_ragas.py

Output:
    eval/reports/baseline-YYYYMMDD.json
"""
from __future__ import annotations

import json
import logging
import sys
import argparse
from datetime import datetime
from pathlib import Path

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.llm.client import get_chat_llm
from app.llm.embedding import embed_text
from app.vector_db import get_milvus_client
from app.vector_db.collections import knowledge as knowledge_vdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent.parent
REPORTS_DIR = EVAL_DIR / "reports"
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"


def load_golden_set(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def run_evaluation(args):
    golden_path = Path(args.golden_set) if args.golden_set else GOLDEN_SET_PATH
    golden = load_golden_set(golden_path)
    entries = golden.get("entries", [])
    logger.info(f"Loaded {len(entries)} golden entries")

    client = get_milvus_client()
    llm = get_chat_llm(temperature=0.0)
    samples = []

    for entry in entries:
        query = entry["query"]
        reference = entry["reference_answer"]

        # Retrieve context using current pipeline (vector only for baseline)
        try:
            query_vec = await embed_text(query)
            chunks = knowledge_vdb.search(
                client=client,
                query_vector=query_vec,
                top_k=4,
            )
            contexts = [c.get("content", "") for c in chunks if c.get("content")]
        except Exception as e:
            logger.warning(f"Retrieval failed for {entry['id']}: {e}")
            contexts = []

        if not contexts:
            logger.warning(f"No contexts retrieved for {entry['id']}, skipping")
            continue

        # Generate answer using LLM with retrieved context
        ctx_text = "\n\n---\n".join(contexts[:4])
        answer_prompt = (
            f"基于以下参考资料回答问题：\n\n"
            f"参考资料：\n{ctx_text}\n\n"
            f"问题：{query}\n\n"
            f"请用中文回答。"
        )
        try:
            response = await llm.ainvoke(answer_prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"LLM generation failed for {entry['id']}: {e}")
            answer = ""

        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        )
        samples.append(sample)

    if not samples:
        logger.error("No valid samples for evaluation")
        return

    logger.info(f"Running RAGAS evaluation on {len(samples)} samples...")

    result = evaluate(
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
        dataset=samples,
    )

    # Build report
    report = {
        "date": datetime.now().isoformat(),
        "golden_set_version": golden.get("version"),
        "num_samples": len(samples),
        "metrics": {},
    }
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision",
                         "context_recall", "answer_correctness"]:
        score = result.get(metric_name, None)
        if score is not None:
            report["metrics"][metric_name] = float(score) if hasattr(score, "__float__") else score

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    report_path = Path(args.output) if args.output else REPORTS_DIR / f"baseline-{date_str}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Report saved to {report_path}")
    logger.info(f"Metrics: {json.dumps(report['metrics'], indent=2)}")

    if args.upload:
        _upload_to_langsmith(report, golden, args.experiment_name)


def parse_args():
    parser = argparse.ArgumentParser(description="RAGAS evaluation script")
    parser.add_argument("--golden-set", default=None, help="Path to golden set JSON (default: eval/golden_set.json)")
    parser.add_argument("--output", default=None, help="Output report path (default: eval/reports/baseline-{date}.json)")
    parser.add_argument("--upload", action="store_true", help="Upload results to LangSmith experiment")
    parser.add_argument("--experiment-name", default=None, help="LangSmith experiment name (default: ragas-{date})")
    return parser.parse_args()


def _upload_to_langsmith(report, golden_data, experiment_name):
    """Upload RAGAS results as LangSmith experiment"""
    import os
    from datetime import date

    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("LANGSMITH_API_KEY not set. Skipping LangSmith upload.")
        return

    from langsmith import Client

    ls_client = Client()
    if experiment_name is None:
        experiment_name = f"ragas-{date.today().isoformat()}"

    dataset_name = "shilian-golden-set"

    # Find dataset
    try:
        dataset = ls_client.read_dataset(dataset_name=dataset_name)
    except Exception:
        print(f"LangSmith dataset '{dataset_name}' not found. Run upload_golden_set.py first.")
        return

    # Upload aggregated metrics as experiment results
    ls_client.create_experiment(
        name=experiment_name,
        dataset_id=dataset.id,
        metadata=report.get("metrics", {}),
    )

    print(f"Uploaded RAGAS results to LangSmith experiment: {experiment_name}")


if __name__ == "__main__":
    import asyncio
    args = parse_args()
    asyncio.run(run_evaluation(args))
